# Synthetic Data & Data-Centric AI: Complete Guide

## Table of Contents
1. [Introduction to Data-Centric AI](#introduction-to-data-centric-ai)
2. [Synthetic Data Generation](#synthetic-data-generation)
3. [Tabular Synthetic Data](#tabular-synthetic-data)
4. [LLM-Generated Training Data](#llm-generated-training-data)
5. [Image Augmentation and Synthesis](#image-augmentation-and-synthesis)
6. [Data Quality and Cleaning](#data-quality-and-cleaning)
7. [Active Learning](#active-learning)
8. [Data Labeling and Annotation](#data-labeling-and-annotation)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)
11. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
12. [Benchmarks and SOTA References](#benchmarks-and-sota-references)
13. [Further Reading](#further-reading)

---

## Introduction to Data-Centric AI

**Data-centric AI** shifts focus from model architecture to data quality. Same model + better data > better model + same data.

### Intuition: Why Data Quality Matters More Than Architecture

Andrew Ng's central thesis: once you have a reasonable architecture, **improving data** typically yields larger gains than tweaking the model. Noisy labels, distribution mismatch, and sparse coverage of edge cases limit even the best architectures. Data-centric approaches—cleaning, augmenting, synthetic generation, and selective labeling—target these bottlenecks directly.

### Model-Centric vs Data-Centric

| Model-Centric | Data-Centric |
|---------------|--------------|
| Fix data, improve model | Fix model, improve data |
| Architecture search | Data cleaning, labeling, augmentation |
| "More data is better" | "Better data is better" |
| SOTA model race | Systematic data quality |

### When Data-Centric Wins

- Labels are noisy or inconsistent
- Limited labeled data
- Domain shift between train and deploy
- Long-tail classes
- Regulatory need for data transparency

---

## Synthetic Data Generation

### Why Synthetic Data?

| Motivation | Example |
|------------|---------|
| **Privacy** | HIPAA: can't share patient data |
| **Scarcity** | Rare diseases, fraud, edge cases |
| **Cost** | Labeling is expensive |
| **Imbalance** | Oversample minority classes |
| **Augmentation** | More diverse training examples |

### Types of Synthetic Data

- **Statistical**: Sample from estimated distribution
- **GAN-based**: Learn to generate realistic data
- **VAE-based**: Decode from latent space
- **LLM-generated**: Text, code, instructions
- **Simulation**: Physics engines, game environments
- **Diffusion-based**: Images, audio

---

## Tabular Synthetic Data

### SMOTE (Oversampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Generates synthetic minority samples by interpolation
```

### CTGAN (Conditional Tabular GAN)

**Intuition**: CTGAN uses a GAN with a conditional generator—given a column value (e.g., class label), it generates coherent rows. Good for tables with mixed types (continuous, discrete, categorical). Use when you need high-fidelity, non-independent synthetic rows.

```python
# pip install sdv
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# 1. Infer schema (column types, constraints)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# 2. Train GAN-based synthesizer (encoder + generator + discriminator)
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=500,
    verbose=True,
    # cuda=True  # Use GPU if available
)
synthesizer.fit(real_data)

# 3. Sample synthetic rows (preserves correlations)
synthetic_data = synthesizer.sample(num_rows=10000)

# 4. Optional: conditional sampling (e.g., oversample minority class)
# synthetic_minority = synthesizer.sample(conditions={"target": 1}, num_rows=5000)
```

### Privacy-Preserving Generation

```python
# Differential privacy + synthetic data
# Ensures no single record can be reverse-engineered

from sdv.single_table import GaussianCopulaSynthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)
# Add noise calibrated to DP budget
```

### Evaluation: Synthetic Data Quality

```python
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

quality = evaluate_quality(real_data, synthetic_data, metadata)
print(f"Quality score: {quality.get_score():.2f}")
# Checks: column shapes, column pair trends, statistical similarity

diagnostic = run_diagnostic(real_data, synthetic_data, metadata)
# Checks: no data copying, valid ranges, coverage
```

---

## LLM-Generated Training Data

### Instruction Following Data

```python
def generate_instruction_data(llm, seed_tasks, n=1000):
    """Generate diverse instruction-response pairs"""
    generated = []
    for i in range(n):
        seed = random.choice(seed_tasks)
        prompt = f"""
        Based on this example instruction:
        Instruction: {seed['instruction']}
        Response: {seed['response']}
        
        Generate a NEW, different instruction-response pair on a related topic.
        Format:
        Instruction: ...
        Response: ...
        """
        result = llm.generate(prompt)
        inst, resp = parse_instruction_response(result)
        generated.append({"instruction": inst, "response": resp})
    return generated
```

### Self-Instruct (Alpaca-Style)

```python
# 1. Start with 175 seed tasks
# 2. LLM generates new tasks based on seeds
# 3. Filter duplicates and low quality
# 4. LLM generates responses for tasks
# 5. Fine-tune on generated data

def self_instruct_pipeline(llm, seeds, n_rounds=10, batch_size=20):
    task_pool = list(seeds)
    for round in range(n_rounds):
        batch_seeds = random.sample(task_pool, min(3, len(task_pool)))
        prompt = build_generation_prompt(batch_seeds)
        new_tasks = llm.generate(prompt, n=batch_size)
        filtered = filter_quality(new_tasks, task_pool)
        task_pool.extend(filtered)
    return task_pool
```

### Text Classification Data

```python
def generate_classification_examples(llm, label, n=100):
    """Generate synthetic labeled text examples"""
    prompt = f"""
    Generate {n} diverse text examples for the label: "{label}"
    Each should be 1-3 sentences. Vary style, topic, and complexity.
    Format: one example per line.
    """
    examples = llm.generate(prompt)
    return [(ex.strip(), label) for ex in examples.split('\n') if ex.strip()]

# Generate for all labels
all_data = []
for label in ["positive", "negative", "neutral"]:
    all_data.extend(generate_classification_examples(llm, label, n=200))
```

### Preference Data Generation

```python
def generate_preference_pairs(llm, prompts):
    """Generate (chosen, rejected) pairs for DPO training"""
    pairs = []
    for prompt in prompts:
        good = llm.generate(prompt, temperature=0.3)  # Higher quality
        bad = llm.generate(prompt, temperature=1.2)  # Lower quality
        pairs.append({"prompt": prompt, "chosen": good, "rejected": bad})
    return pairs
```

---

## Image Augmentation and Synthesis

### Classical Augmentation

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(15),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Mixup and CutMix

```python
def mixup(images, labels, alpha=0.2):
    """Mixup: interpolate between pairs"""
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0))
    mixed_images = lam * images + (1 - lam) * images[indices]
    return mixed_images, labels, labels[indices], lam

def rand_bbox(size, lam):
    """Random bounding box for CutMix; lam controls area ratio."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images, labels, alpha=1.0):
    """CutMix: paste patch from one image onto another"""
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0))
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape, lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2-bbx1)*(bby2-bby1)) / (images.shape[-1]*images.shape[-2])
    return images, labels, labels[indices], lam
```

### Diffusion-Based Image Generation

**Intuition**: Diffusion models learn to denoise; sampling runs the reverse process. For augmentation, they generate novel images conditioned on class labels or text. Unlike GANs, diffusion avoids mode collapse and produces diverse, high-fidelity samples.

```python
from diffusers import StableDiffusionPipeline
import torch
import os

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate training images with diversity via seeds and prompt variations
def generate_synthetic_dataset(labels, n_per_class=100, prompt_templates=None):
    """Generate class-balanced synthetic images with prompt diversity."""
    if prompt_templates is None:
        prompt_templates = ["a photo of a {}", "{} in natural lighting", "{} close-up"]
    for label in labels:
        os.makedirs(f"synthetic/{label}", exist_ok=True)
        for i in range(n_per_class):
            template = prompt_templates[i % len(prompt_templates)]
            prompt = template.format(label)
            # Vary seed for diversity
            generator = torch.Generator(device="cuda").manual_seed(42 + i)
            image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
            image.save(f"synthetic/{label}/{i}.png")
```

### Diffusion-Based Augmentation (Img2Img)

Use diffusion to *transform* existing images instead of generating from scratch—preserves identity while adding variation.

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# Strength 0.3–0.5: subtle variation; 0.7+: more creative changes
def augment_with_diffusion(image, prompt="same style, slight variation", strength=0.4):
    return pipe(prompt=prompt, image=image, strength=strength).images[0]
```

---

## Data Quality and Cleaning

### Detecting Label Noise

```python
# Confident Learning (cleanlab)
# pip install cleanlab
from cleanlab.classification import CleanLearning

cl = CleanLearning(clf=sklearn_model)
label_issues = cl.find_label_issues(X, y)
# Returns indices of likely mislabeled examples

# Fix: Re-label, remove, or down-weight
clean_mask = ~label_issues["is_label_issue"]
X_clean, y_clean = X[clean_mask], y[clean_mask]
```

### Data Profiling

```python
# pandas-profiling / ydata-profiling
from ydata_profiling import ProfileReport

report = ProfileReport(df, title="Data Quality Report")
report.to_file("report.html")
# Shows: distributions, missing values, correlations, duplicates
```

### Outlier Detection in Data

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.05)
outlier_labels = clf.fit_predict(X)
clean_data = X[outlier_labels == 1]
```

### Deduplication

```python
# Exact dedup
df_dedup = df.drop_duplicates()

# Near-duplicate (text): MinHash
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.8, num_perm=128)
# Index and query for near-duplicates
```

---

## Active Learning

**Active learning**: Model selects the most informative unlabeled samples for human labeling. Goal: achieve target accuracy with fewer labels than random sampling.

### Intuition: When Does Active Learning Help?

Active learning works best when (1) the model's uncertainty correlates with *true* informativeness, and (2) the unlabeled pool is **representative** of the target distribution. If the pool is biased or out-of-distribution, selecting by uncertainty can reinforce the bias. Use **diversity** (e.g., clustering) alongside uncertainty to avoid this.

### Pool-Based vs Stream-Based

| Mode | When to Use |
|------|-------------|
| **Pool-based** | Fixed unlabeled set; can score all before selecting |
| **Stream-based** | Data arrives sequentially; must decide per example |
| **Synthetic** | Use active learning to choose *which* synthetic samples to add |

### Advanced Strategies

1. **Uncertainty sampling**: Highest entropy, or lowest max-prob (least confident)
2. **Margin sampling**: Smallest gap between top-2 class probabilities
3. **Query-by-committee (QBC)**: Train K models; label where they disagree most
4. **Expected model change**: Choose samples that would change the model most if labeled
5. **Diversity + uncertainty**: Cluster embeddings; within each cluster, pick most uncertain (balances exploration vs exploitation)

### Implementation

```python
import numpy as np

def uncertainty_sampling(model, unlabeled_pool, n_samples=100, strategy="entropy"):
    """
    Select most uncertain samples for labeling.
    strategy: 'entropy' | 'least_confident' | 'margin'
    """
    probs = model.predict_proba(unlabeled_pool)
    eps = 1e-10
    
    if strategy == "entropy":
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        top_indices = np.argsort(entropy)[-n_samples:]
    elif strategy == "least_confident":
        max_probs = probs.max(axis=1)
        top_indices = np.argsort(max_probs)[:n_samples]  # Lowest confidence
    elif strategy == "margin":
        sorted_probs = np.sort(probs, axis=1)[:, -2:]  # Top 2
        margin = sorted_probs[:, 1] - sorted_probs[:, 0]
        top_indices = np.argsort(margin)[:n_samples]  # Smallest margin
    return top_indices

def diversity_aware_sampling(model, unlabeled_pool, embeddings, n_samples=100, n_clusters=20):
    """Combine diversity (cluster) with uncertainty within clusters."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    probs = model.predict_proba(unlabeled_pool)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    selected = []
    per_cluster = max(1, n_samples // n_clusters)
    for c in range(n_clusters):
        mask = kmeans.labels_ == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        top_local = np.argsort(entropy[mask])[-per_cluster:][::-1]
        selected.extend(indices[top_local[:per_cluster]])
    return np.array(selected[:n_samples])

# Active learning loop
labeled_X, labeled_y = initial_labeled_set()
unlabeled_pool = get_unlabeled_pool()
for round in range(10):
    model.fit(labeled_X, labeled_y)
    indices = uncertainty_sampling(model, unlabeled_pool, n_samples=50)
    new_labels = human_label(unlabeled_pool[indices])
    labeled_X = np.vstack([labeled_X, unlabeled_pool[indices]])
    labeled_y = np.concatenate([labeled_y, new_labels])
    unlabeled_pool = np.delete(unlabeled_pool, indices, axis=0)
```

### LLM-Assisted Labeling

```python
def llm_label(text, label_options):
    prompt = f"""
    Classify: "{text}"
    Options: {label_options}
    Return only the label.
    """
    return llm.generate(prompt).strip()

# Use LLM for initial labels, human for corrections
```

---

## Data Labeling and Annotation

### Label Studio (Open Source)

```python
# pip install label-studio
# label-studio start
# Web UI for text, image, audio labeling
# Supports: classification, NER, bounding boxes, segmentation
```

### Programmatic Labeling (Snorkel)

```python
# Define labeling functions
from snorkel.labeling import labeling_function

@labeling_function()
def lf_keyword_positive(x):
    return 1 if any(w in x.text.lower() for w in ["great", "awesome", "love"]) else -1

@labeling_function()
def lf_keyword_negative(x):
    return 0 if any(w in x.text.lower() for w in ["terrible", "hate", "awful"]) else -1

# Combine with label model
from snorkel.labeling.model import LabelModel
label_model = LabelModel(cardinality=2)
label_model.fit(L_train)
probabilistic_labels = label_model.predict_proba(L_train)
```

### Annotation Guidelines

- **Clear definitions**: Concrete examples per category
- **Edge cases**: Document and resolve ambiguities
- **Inter-annotator agreement**: Measure Cohen's kappa
- **Iterative**: Refine guidelines based on disagreements

---

## Practical Examples

### Example 1: Synthetic Tabular Data Pipeline

```python
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# Fit
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_df)
synth = CTGANSynthesizer(metadata, epochs=300)
synth.fit(real_df)

# Generate
synthetic_df = synth.sample(num_rows=5000)

# Validate
quality = evaluate_quality(real_df, synthetic_df, metadata)
print(f"Quality: {quality.get_score():.2f}")
```

### Example 2: Data Cleaning with Cleanlab

```python
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression

cl = CleanLearning(clf=LogisticRegression())
cl.fit(X_train, y_train_noisy)
# Automatically finds and handles label errors
preds = cl.predict(X_test)
```

### Example 3: LLM Data Generation for NER

```python
def generate_ner_examples(llm, entity_types, n=100):
    prompt = f"""
    Generate {n} sentences with the following named entities marked:
    Entity types: {entity_types}
    
    Format: "sentence" | entity1:TYPE, entity2:TYPE
    Example: "Apple released iPhone in Cupertino" | Apple:ORG, iPhone:PRODUCT, Cupertino:LOC
    """
    return parse_ner_examples(llm.generate(prompt))
```

---

## Best Practices

1. **Validate synthetic data** against real distribution
2. **Mix synthetic + real** for best results
3. **Clean before augment**: Fix labels before generating more
4. **Active learning**: Label efficiently, not exhaustively
5. **Privacy**: Validate synthetic data doesn't memorize real records
6. **Document lineage**: Track data sources and transformations

---

## Summary

| Technique | Use Case | Tool |
|-----------|----------|------|
| CTGAN | Tabular privacy-safe | SDV |
| LLM generation | Text, instructions | OpenAI, open LLMs |
| Diffusion | Images | Stable Diffusion |
| Cleanlab | Label noise | cleanlab |
| Active Learning | Efficient labeling | modAL, custom |
| Snorkel | Programmatic labels | snorkel |

**Libraries**: `sdv`, `cleanlab`, `snorkel`, `imblearn`, `label-studio`, `diffusers`
