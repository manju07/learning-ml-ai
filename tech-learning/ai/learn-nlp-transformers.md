# Natural Language Processing & Transformers: A Complete Deep-Dive

## Table of Contents
1. [Introduction to NLP](#1-introduction-to-nlp)
2. [Text Preprocessing Pipeline](#2-text-preprocessing-pipeline)
3. [Classical NLP: Statistical Methods](#3-classical-nlp-statistical-methods)
4. [Word Embeddings](#4-word-embeddings)
5. [Sequence Models: RNN, LSTM, GRU](#5-sequence-models-rnn-lstm-gru)
6. [Attention Mechanism](#6-attention-mechanism)
7. [The Transformer Architecture](#7-the-transformer-architecture)
8. [Tokenizers In Depth: BPE, WordPiece, SentencePiece](#8-tokenizers-in-depth)
9. [BERT and Encoder-Only Models](#9-bert-and-encoder-only-models)
10. [GPT and Decoder-Only Models](#10-gpt-and-decoder-only-models)
11. [Seq2Seq Models: T5 and BART](#11-seq2seq-models-t5-and-bart)
12. [Sentence Transformers and SBERT](#12-sentence-transformers-and-sbert)
13. [NLP Tasks: NER, POS, Classification](#13-nlp-tasks)
14. [Machine Translation and Summarization](#14-machine-translation-and-summarization)
15. [Question Answering](#15-question-answering)
16. [Modern Positional Encodings](#16-modern-positional-encodings)
17. [Advanced Techniques: Flash Attention and Efficiency](#17-advanced-techniques)

---

## 1. Introduction to NLP

Natural Language Processing (NLP) is a subfield of artificial intelligence focused on enabling computers to understand, interpret, generate, and reason about human language. It sits at the intersection of linguistics, computer science, and machine learning.

### 1.1 The NLP Task Landscape

| Task | Description | Example |
|------|-------------|---------|
| Text Classification | Assign label(s) to text | Spam detection, sentiment |
| Named Entity Recognition | Tag entities (persons, orgs, locations) | "Apple [ORG] hired Tim Cook [PER]" |
| Part-of-Speech Tagging | Assign grammatical tags | "runs" → VERB |
| Dependency Parsing | Identify grammatical structure | Subject-verb-object relationships |
| Machine Translation | Translate between languages | EN → FR |
| Summarization | Condense long text | Abstractive or extractive |
| Question Answering | Answer from context | Extractive (span) or generative |
| Text Generation | Produce coherent text | Story, code, chat |
| Coreference Resolution | Link mentions to entities | "he" → "John Smith" |
| Relation Extraction | Find relations between entities | "founded_by(Apple, Jobs)" |

### 1.2 Evolution of NLP

```
1950s-1980s: Rule-based systems (Expert systems, hand-crafted grammars)
    ↓
1990s-2000s: Statistical NLP (HMM, CRF, N-grams, SVM)
    ↓
2010-2017: Neural NLP (Word2Vec, RNNs, LSTMs, CNNs for text)
    ↓
2017-2018: Attention & Transformers ("Attention is All You Need")
    ↓
2018-2020: Pre-trained language models (BERT, GPT-2, XLNet, RoBERTa)
    ↓
2020-present: Large Language Models (GPT-3/4, LLaMA, Mistral, Claude)
```

---

## 2. Text Preprocessing Pipeline

### 2.1 Why Preprocessing Matters

Raw text is noisy — it contains HTML tags, URLs, special characters, inconsistent casing, and linguistic noise. Preprocessing normalizes text to reduce vocabulary size and improve downstream model performance.

### 2.2 Tokenization

Tokenization splits text into units (tokens). Tokens can be words, subwords, characters, or bytes.

**Word-level tokenization:**
```python
import re
from typing import List

def word_tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    # Split on whitespace and punctuation boundaries
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return tokens

text = "I'm learning NLP! It's fascinating."
print(word_tokenize_simple(text))
# ['I', "'", 'm', 'learning', 'NLP', '!', 'It', "'", 's', 'fascinating', '.']
```

**NLTK tokenization:**
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.tokenize import MWETokenizer  # Multi-word expression tokenizer

# Download once
# nltk.download('punkt')
# nltk.download('punkt_tab')

text = "Dr. Smith went to Washington D.C. He met the president."

# Word tokenization (handles abbreviations)
words = word_tokenize(text)
print("Words:", words)
# ['Dr.', 'Smith', 'went', 'to', 'Washington', 'D.C.', 'He', 'met', 'the', 'president', '.']

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
# ['Dr. Smith went to Washington D.C.', 'He met the president.']

# Tweet tokenizer (handles hashtags, emoji)
tweet_tokenizer = TweetTokenizer()
tweet = "I love #NLP 😊 @research https://arxiv.org"
print(tweet_tokenizer.tokenize(tweet))
```

### 2.3 Stemming vs. Lemmatization

**Stemming** applies heuristic rules to chop word endings. Fast but linguistically rough.

**Lemmatization** maps words to their dictionary base form (lemma) using linguistic knowledge.

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

words = ["running", "runs", "ran", "better", "good", "studies", "studying", "flies"]

# Stemming with Porter Stemmer
porter = PorterStemmer()
print("Porter:", [porter.stem(w) for w in words])
# ['run', 'run', 'ran', 'better', 'good', 'studi', 'studi', 'fli']

# Snowball (more aggressive, multilingual)
snowball = SnowballStemmer("english")
print("Snowball:", [snowball.stem(w) for w in words])

# Lemmatization with WordNet
lemmatizer = WordNetLemmatizer()
# Must specify POS for accuracy
print("Lemma (v):", lemmatizer.lemmatize("running", pos='v'))   # run
print("Lemma (a):", lemmatizer.lemmatize("better", pos='a'))    # good
print("Lemma (n):", lemmatizer.lemmatize("studies", pos='n'))   # study

# POS-aware lemmatization
from nltk import pos_tag

def get_wordnet_pos(treebank_tag: str) -> str:
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_with_pos(text: str) -> List[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return lemmas

print(lemmatize_with_pos("The dogs are running faster than the cats"))
# ['the', 'dog', 'be', 'run', 'fast', 'than', 'the', 'cat']
```

### 2.4 Stopword Removal

Stopwords are high-frequency, low-information words. Removing them reduces noise in bag-of-words models (but NOT for transformers — they need all context).

```python
from nltk.corpus import stopwords
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
print(f"Total English stopwords: {len(stop_words)}")

# Custom stopwords
custom_stops = stop_words.union({"etc", "also", "would", "could"})

def remove_stopwords(tokens: List[str], stops: set = stop_words) -> List[str]:
    return [t for t in tokens if t.lower() not in stops and len(t) > 1]

tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
filtered = remove_stopwords(tokens)
print(filtered)  # ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

### 2.5 Full Preprocessing Pipeline

```python
import re
import string
import unicodedata
from typing import Optional

class TextPreprocessor:
    """Production-grade text preprocessing pipeline."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatize: bool = True,
        min_token_len: int = 2
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatize = lemmatize
        self.min_token_len = min_token_len
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    def remove_url_patterns(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    def remove_html_tags(self, text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)
    
    def preprocess(self, text: str) -> str:
        # Unicode normalization
        text = self.normalize_unicode(text)
        
        # Remove HTML
        if self.remove_html:
            text = self.remove_html_tags(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.remove_url_patterns(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation tokens
        if self.remove_punctuation:
            tokens = [t for t in tokens if t not in string.punctuation]
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= self.min_token_len]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stemming or Lemmatization (mutually exclusive)
        if self.stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.lemmatize:
            tagged = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in tagged]
        
        return ' '.join(tokens)

# Usage
preprocessor = TextPreprocessor()
text = """<p>Machine learning is <b>amazing</b>! Visit https://example.com for more.
The algorithms are running and learning from massive datasets.</p>"""
print(preprocessor.preprocess(text))
# 'machine learning amazing algorithm run learn massive dataset'
```

### 2.6 spaCy for Advanced NLP

```python
import spacy

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
doc = nlp(text)

# Token-level information
for token in doc:
    print(f"{token.text:<20} {token.lemma_:<15} {token.pos_:<8} {token.dep_:<12} {token.is_stop}")

print("\n--- Named Entities ---")
for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_:<10} {spacy.explain(ent.label_)}")
# Apple Inc.               ORG        Companies, agencies, institutions
# Steve Jobs               PERSON     People
# Cupertino                GPE        Geo-political entity (city, state, country)
# California               GPE
# 1976                     DATE

# Dependency parsing
print("\n--- Dependency Parse ---")
for token in doc:
    print(f"{token.text:<15} --[{token.dep_}]--> {token.head.text}")

# Noun chunks
print("\n--- Noun Chunks ---")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' (root: {chunk.root.text})")
```

---

## 3. Classical NLP: Statistical Methods

### 3.1 Bag of Words (BoW)

BoW represents text as an unordered collection of word frequencies, ignoring grammar and order.

**Mathematical formulation:**
Given vocabulary \( V = \{w_1, w_2, \ldots, w_{|V|}\} \) and document \( d \):

\[ \text{BoW}(d) = [c(w_1, d),\ c(w_2, d),\ \ldots,\ c(w_{|V|}, d)] \]

where \( c(w_i, d) \) is the count of word \( w_i \) in document \( d \).

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love deep learning",
    "Deep learning uses neural networks"
]

# Basic BoW
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()

print("Vocabulary:", vocab)
print("BoW matrix shape:", X.shape)
print("\nBoW matrix:\n", X.toarray())

# With n-grams and min_df filtering
vectorizer_ngram = CountVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=1000,
    min_df=2,            # Must appear in at least 2 docs
    max_df=0.95          # Ignore terms in >95% docs (stopwords)
)
X_ngram = vectorizer_ngram.fit_transform(corpus)
print("\nN-gram features:", vectorizer_ngram.get_feature_names_out())
```

### 3.2 TF-IDF: Term Frequency-Inverse Document Frequency

TF-IDF weights terms by their importance — common terms in a document but rare across the corpus score highest.

**Mathematical formulation:**

\[
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
\]

\[
\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1
\]

\[
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
\]

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1,
    sublinear_tf=True  # Replace TF with 1 + log(TF)
)
X_tfidf = tfidf.fit_transform(corpus)

# Document similarity using cosine similarity
doc_similarities = cosine_similarity(X_tfidf)
print("Document similarity matrix:")
print(np.round(doc_similarities, 3))

# Find most similar document to a query
def find_similar_docs(query: str, tfidf, corpus, top_k: int = 3):
    query_vec = tfidf.transform([query])
    similarities = cosine_similarity(query_vec, X_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(corpus[i], similarities[i]) for i in top_indices]

results = find_similar_docs("neural networks", tfidf, corpus)
for doc, score in results:
    print(f"  Score {score:.3f}: {doc}")

# Manual TF-IDF implementation for understanding
class TFIDFManual:
    """Manual TF-IDF for educational purposes."""
    
    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        from collections import Counter
        import math
        
        tokenized = [doc.lower().split() for doc in corpus]
        
        # Build vocabulary
        all_words = set(word for doc in tokenized for word in doc)
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        N = len(corpus)
        
        # Compute IDF
        self.idf = {}
        for word in self.vocab:
            df = sum(1 for doc in tokenized if word in doc)
            self.idf[word] = math.log((1 + N) / (1 + df)) + 1
        
        # Compute TF-IDF matrix
        matrix = np.zeros((N, len(self.vocab)))
        for i, doc in enumerate(tokenized):
            tf = Counter(doc)
            total = len(doc)
            for word, count in tf.items():
                if word in self.vocab:
                    j = self.vocab[word]
                    matrix[i, j] = (count / total) * self.idf[word]
        
        # L2 normalization
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-10)
```

### 3.3 N-grams and Language Models

An **N-gram** is a contiguous sequence of N items (words, characters, bytes).

**N-gram language model:**

\[
P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_{i-N+1}, \ldots, w_{i-1})
\]

With **Laplace (add-1) smoothing:**

\[
P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
\]

```python
from collections import Counter, defaultdict
import math
from typing import Dict, Tuple

class NGramLanguageModel:
    """Bigram language model with Laplace smoothing."""
    
    def __init__(self, n: int = 2, smoothing: float = 1.0):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts: Dict = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set = set()
    
    def tokenize(self, text: str) -> List[str]:
        tokens = ['<s>'] * (self.n - 1) + text.lower().split() + ['</s>']
        return tokens
    
    def train(self, corpus: List[str]):
        for text in corpus:
            tokens = self.tokenize(text)
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
    
    def log_prob(self, word: str, context: tuple) -> float:
        """Compute log P(word | context) with Laplace smoothing."""
        count = self.ngram_counts[context][word] + self.smoothing
        total = self.context_counts[context] + self.smoothing * len(self.vocab)
        return math.log(count / total)
    
    def sentence_log_prob(self, sentence: str) -> float:
        tokens = self.tokenize(sentence)
        log_p = 0.0
        for i in range(self.n-1, len(tokens)):
            context = tuple(tokens[i-self.n+1:i])
            word = tokens[i]
            log_p += self.log_prob(word, context)
        return log_p
    
    def perplexity(self, test_corpus: List[str]) -> float:
        total_log_p = 0.0
        total_tokens = 0
        for sent in test_corpus:
            total_log_p += self.sentence_log_prob(sent)
            total_tokens += len(sent.split())
        return math.exp(-total_log_p / total_tokens)
    
    def generate(self, max_tokens: int = 20) -> str:
        import random
        context = tuple(['<s>'] * (self.n - 1))
        words = []
        for _ in range(max_tokens):
            candidates = self.ngram_counts.get(context, {})
            if not candidates or '</s>' in candidates:
                break
            # Sample proportionally to counts
            total = sum(candidates.values())
            r = random.random() * total
            cumulative = 0
            for word, count in candidates.items():
                cumulative += count
                if r <= cumulative:
                    words.append(word)
                    context = tuple(list(context[1:]) + [word])
                    break
        return ' '.join(words)

# Train
lm = NGramLanguageModel(n=2)
lm.train(corpus)
print("Sentence probability:", lm.sentence_log_prob("machine learning is great"))
print("Perplexity:", lm.perplexity(["machine learning"]))
print("Generated:", lm.generate())
```

### 3.4 Word Co-occurrence Matrix and PMI

The **Pointwise Mutual Information (PMI)** measures word association:

\[
\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}
\]

Positive PMI (PPMI) sets negative values to 0, giving better embeddings:

```python
import numpy as np
from scipy.sparse import lil_matrix

def build_cooccurrence_matrix(corpus: List[str], window: int = 2, vocab_size: int = 1000):
    """Build word co-occurrence matrix."""
    from collections import Counter
    
    # Build vocabulary
    all_words = ' '.join(corpus).lower().split()
    word_freq = Counter(all_words)
    vocab = [w for w, _ in word_freq.most_common(vocab_size)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    n = len(vocab)
    cooc = np.zeros((n, n))
    
    for text in corpus:
        tokens = text.lower().split()
        for i, word in enumerate(tokens):
            if word not in word2idx:
                continue
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if i != j and tokens[j] in word2idx:
                    cooc[word2idx[word], word2idx[tokens[j]]] += 1
    
    # Compute PPMI
    total = cooc.sum()
    row_sums = cooc.sum(axis=1, keepdims=True)
    col_sums = cooc.sum(axis=0, keepdims=True)
    
    # P(w1, w2) / (P(w1) * P(w2))
    with np.errstate(divide='ignore', invalid='ignore'):
        ppmi = np.log2(cooc * total / (row_sums @ col_sums + 1e-10))
    ppmi = np.maximum(ppmi, 0)  # PPMI
    
    return ppmi, vocab, word2idx
```

---

## 4. Word Embeddings

Word embeddings map discrete tokens to dense continuous vectors in \( \mathbb{R}^d \), capturing semantic relationships.

### 4.1 Word2Vec: Skip-Gram and CBOW

**Word2Vec** (Mikolov et al., 2013) learns embeddings by predicting context from word (Skip-gram) or word from context (CBOW).

**Skip-gram objective:**

Given a center word \( w_c \) and context words \( w_{c-m}, \ldots, w_{c+m} \) (window size \( m \)):

\[
\mathcal{L} = -\sum_{(c, o) \in \text{pairs}} \log P(w_o \mid w_c)
\]

\[
P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}
\]

The softmax over entire vocabulary is expensive. **Negative Sampling** approximates it:

\[
\mathcal{L}_{\text{NS}} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})]
\]

where \( P_n(w) \propto f(w)^{3/4} \) is the noise distribution (unigram distribution raised to 3/4 power).

**CBOW objective:**

Predict center word from average of context embeddings:

\[
\hat{\mathbf{v}} = \frac{1}{2m} \sum_{j \neq 0, -m \leq j \leq m} \mathbf{v}_{w_{c+j}}
\]

\[
P(w_c \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_c}^\top \hat{\mathbf{v}})}{\sum_{w} \exp(\mathbf{u}_w^\top \hat{\mathbf{v}})}
\]

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

# Prepare corpus (list of tokenized sentences)
sentences = [
    ["machine", "learning", "is", "a", "field", "of", "ai"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["nlp", "is", "natural", "language", "processing"],
    ["transformers", "revolutionized", "nlp"],
    ["bert", "gpt", "are", "transformer", "models"],
]

# Skip-gram model
skip_gram = Word2Vec(
    sentences,
    vector_size=100,    # Embedding dimension d
    window=5,           # Context window size
    min_count=1,        # Minimum word frequency
    sg=1,               # 1=Skip-gram, 0=CBOW
    negative=10,        # Number of negative samples
    ns_exponent=0.75,   # Noise distribution exponent
    alpha=0.025,        # Initial learning rate
    epochs=10,
    workers=4
)

# CBOW model
cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Word vectors
v_learning = skip_gram.wv['learning']
print(f"'learning' vector shape: {v_learning.shape}")  # (100,)
print(f"'learning' vector[:5]: {v_learning[:5]}")

# Semantic similarity
similarity = skip_gram.wv.similarity('machine', 'deep')
print(f"Similarity(machine, deep): {similarity:.4f}")

# Most similar words
print("\nMost similar to 'learning':")
for word, score in skip_gram.wv.most_similar('learning', topn=5):
    print(f"  {word}: {score:.4f}")

# Word analogy: king - man + woman ≈ queen
# In our toy corpus: machine - learning + nlp ≈ ?
# Works better with large pre-trained models

# Load pre-trained Word2Vec
# w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# print(w2v.most_similar(positive=['king', 'woman'], negative=['man']))  # → queen

# Implement Skip-gram from scratch (educational)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SkipGramDataset(Dataset):
    def __init__(self, sentences, word2idx, window=2):
        self.pairs = []
        for sentence in sentences:
            indices = [word2idx.get(w, 0) for w in sentence]
            for i, center in enumerate(indices):
                for j in range(max(0, i-window), min(len(indices), i+window+1)):
                    if i != j:
                        self.pairs.append((center, indices[j]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # v_c
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # u_o
        
        # Initialize uniformly
        nn.init.uniform_(self.in_embed.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.out_embed.weight)
    
    def forward(self, center, context):
        # center: (B,), context: (B,)
        v_c = self.in_embed(center)   # (B, D)
        u_o = self.out_embed(context) # (B, D)
        return (v_c * u_o).sum(dim=1) # (B,)  dot product
```

### 4.2 GloVe: Global Vectors for Word Representation

**GloVe** (Pennington et al., 2014) combines global co-occurrence statistics with local context.

**Objective:**

\[
J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( \mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
\]

where:
- \( X_{ij} \) = co-occurrence count of words \( i \) and \( j \)
- \( f(x) \) = weighting function: \( \min\!\left(1, \left(\frac{x}{x_{\max}}\right)^\alpha\right) \) with \( \alpha = 0.75, x_{\max} = 100 \)
- \( \mathbf{w}_i, \tilde{\mathbf{w}}_j \) = word and context vectors
- Final embedding: \( \mathbf{w}_i + \tilde{\mathbf{w}}_i \)

```python
import numpy as np

def load_glove(path: str, embedding_dim: int = 100) -> dict:
    """Load pre-trained GloVe embeddings."""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

# Usage:
# glove = load_glove('glove.6B.100d.txt', embedding_dim=100)
# print(glove['king'] - glove['man'] + glove['woman'])  # ≈ glove['queen']

def build_embedding_matrix(vocab: List[str], embeddings: dict, dim: int = 100) -> np.ndarray:
    """Build embedding matrix for use in neural networks."""
    matrix = np.zeros((len(vocab), dim))
    oov_count = 0
    for i, word in enumerate(vocab):
        if word in embeddings:
            matrix[i] = embeddings[word]
        else:
            # Random initialization for OOV words
            matrix[i] = np.random.normal(scale=0.6, size=(dim,))
            oov_count += 1
    print(f"OOV words: {oov_count}/{len(vocab)}")
    return matrix
```

### 4.3 FastText: Subword Embeddings

**FastText** (Bojanowski et al., 2017) represents words as bags of character n-grams. This handles morphologically rich languages and out-of-vocabulary words.

For word "where" with n=3,4,5: `<wh, whe, her, ere, re>, <whe, wher, here, ere>, <wher, where>, <where>` plus the whole word `<where>`.

\[
\mathbf{v}_w = \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g
\]

where \( G_w \) is the set of n-grams for word \( w \), and \( \mathbf{z}_g \) is the embedding for n-gram \( g \).

```python
from gensim.models import FastText

# Train FastText
ft_model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,           # Skip-gram
    min_n=3,        # Minimum n-gram length
    max_n=6,        # Maximum n-gram length
    epochs=10
)

# Works on unseen words!
print(ft_model.wv['machinelearning'])  # Composed from subwords
print(ft_model.wv['transformerz'])     # OOV - subwords still work

# Compare Word2Vec vs FastText on OOV
# Word2Vec would raise KeyError; FastText returns subword-based embedding
```

---

## 5. Sequence Models: RNN, LSTM, GRU

### 5.1 Recurrent Neural Networks (RNN)

RNNs process sequences by maintaining a hidden state:

\[
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
\]

\[
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
\]

**Problem**: Vanishing gradients for long sequences. The gradient of the loss at step \( T \) w.r.t. step \( t \) involves:

\[
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^{T} \text{diag}(\tanh'(\cdot)) \mathbf{W}_{hh}
\]

When eigenvalues of \( W_{hh} < 1 \), gradients vanish exponentially.

### 5.2 Long Short-Term Memory (LSTM)

LSTMs (Hochreiter & Schmidhuber, 1997) use gating mechanisms to selectively remember/forget:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget gate)}
\]

\[
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input gate)}
\]

\[
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \quad \text{(Candidate cell)}
\]

\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Cell state update)}
\]

\[
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output gate)}
\]

\[
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden state)}
\]

### 5.3 Gated Recurrent Unit (GRU)

GRU (Cho et al., 2014) simplifies LSTM with 2 gates:

\[
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Update gate)}
\]

\[
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Reset gate)}
\]

\[
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])
\]

\[
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\]

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """LSTM text classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * directions, num_classes)
    
    def forward(self, x, lengths=None):
        # x: (B, T) token ids
        embedded = self.dropout(self.embedding(x))  # (B, T, D)
        
        if lengths is not None:
            # Pack padded sequences for efficiency
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, (h_n, c_n) = self.lstm(embedded)  # h_n: (num_layers*dirs, B, H)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Use final hidden state from both directions
        if self.lstm.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h_final = h_n[-1]  # (B, H)
        
        return self.fc(self.dropout(h_final))

# Seq2Seq with attention
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Combine directions
    
    def forward(self, src):
        embedded = self.embedding(src)  # (B, T, D)
        outputs, (h, c) = self.lstm(embedded)
        # outputs: (B, T, 2H) — all timesteps
        # Combine forward and backward for initial decoder state
        h = torch.tanh(self.fc(torch.cat([h[-2], h[-1]], dim=1)))
        return outputs, h
```

---

## 6. Attention Mechanism

### 6.1 Bahdanau Attention (Additive Attention)

Bahdanau et al. (2015) introduced attention to allow the decoder to look at all encoder hidden states:

\[
e_{ij} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{h}_i + \mathbf{U}_a \mathbf{s}_{j-1})
\]

\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
\]

\[
\mathbf{c}_j = \sum_i \alpha_{ij} \mathbf{h}_i
\]

### 6.2 Luong Attention (Multiplicative/Dot-product Attention)

\[
e_{ij} = \mathbf{s}_j^\top \mathbf{h}_i \quad \text{(dot)} \quad \text{or} \quad \mathbf{s}_j^\top \mathbf{W}_a \mathbf{h}_i \quad \text{(general)}
\]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BahdanauAttention(nn.Module):
    """Additive attention (Bahdanau et al., 2015)."""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.W_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (B, T, H_enc)
        # decoder_hidden: (B, H_dec)
        
        energy = self.v_a(torch.tanh(
            self.W_a(encoder_outputs) +                    # (B, T, A)
            self.U_a(decoder_hidden).unsqueeze(1)          # (B, 1, A)
        )).squeeze(-1)  # (B, T)
        
        attn_weights = F.softmax(energy, dim=1)  # (B, T)
        context = (attn_weights.unsqueeze(2) * encoder_outputs).sum(dim=1)  # (B, H_enc)
        return context, attn_weights


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (Vaswani et al., 2017)."""
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: (B, heads, T_q, d_k)
        K: (B, heads, T_k, d_k)
        V: (B, heads, T_v, d_v)
        mask: (B, 1, 1, T_k) or (B, 1, T_q, T_k)
        """
        d_k = Q.size(-1)
        
        # Attention scores: (B, heads, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # (B, heads, T_q, d_v)
        return output, attn_weights
```

---

## 7. The Transformer Architecture

The Transformer (Vaswani et al., 2017) completely replaces recurrence with self-attention, enabling full parallelization.

### 7.1 High-Level Architecture

```
Input Sequence → Embedding + Positional Encoding
                        ↓
             [Encoder Stack × N]
              ┌────────────────────┐
              │ Multi-Head Self-   │
              │ Attention          │
              │ + Add & Norm       │
              │                    │
              │ Feed-Forward       │
              │ Network            │
              │ + Add & Norm       │
              └────────────────────┘
                        ↓
             [Decoder Stack × N]
              ┌────────────────────┐
              │ Masked Multi-Head  │
              │ Self-Attention     │
              │ + Add & Norm       │
              │                    │
              │ Cross-Attention    │
              │ (Q=decoder,        │
              │  K,V=encoder)      │
              │ + Add & Norm       │
              │                    │
              │ Feed-Forward       │
              │ + Add & Norm       │
              └────────────────────┘
                        ↓
              Linear + Softmax → Output Tokens
```

### 7.2 Multi-Head Self-Attention: Complete Derivation

Given input \( X \in \mathbb{R}^{n \times d_{\text{model}}} \), for head \( i \):

\[
\mathbf{Q}_i = X \mathbf{W}_i^Q, \quad \mathbf{K}_i = X \mathbf{W}_i^K, \quad \mathbf{V}_i = X \mathbf{W}_i^V
\]

where \( \mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \), \( \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v} \), and \( d_k = d_v = d_{\text{model}} / h \).

\[
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right) \mathbf{V}_i
\]

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O
\]

where \( \mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}} \).

**Why scale by \( \sqrt{d_k} \)?**

The dot product \( Q \cdot K^\top \) grows in magnitude as \( d_k \) increases. With large values, softmax saturates, producing near-zero gradients. Scaling prevents this:

\[
\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i) = d_k \quad \text{(for unit-variance components)}
\]

Dividing by \( \sqrt{d_k} \) restores unit variance.

### 7.3 Positional Encoding

Since self-attention is permutation-equivariant, we add position information via sinusoidal encoding:

\[
\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

\[
\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

This encoding has the property that \( \text{PE}(pos + k) \) can be represented as a linear function of \( \text{PE}(pos) \), allowing the model to generalize to relative positions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Precompute PE table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len, 1)
        
        # Compute division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)  # Not a parameter, but saved in state_dict
    
    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    def visualize(self):
        import matplotlib.pyplot as plt
        pe = self.pe.squeeze(0).numpy()
        plt.figure(figsize=(15, 5))
        plt.imshow(pe.T, aspect='auto', cmap='RdBu')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Positional Encoding')
        plt.colorbar()
        plt.show()


class MultiHeadAttention(nn.Module):
    """Complete multi-head attention implementation."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # All projection matrices in single weight matrices for efficiency
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, d_k)."""
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, heads, T, d_k)
    
    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.size()
        
        # Project to Q, K, V and split into heads
        Q = self.split_heads(self.W_q(query))  # (B, h, T_q, d_k)
        K = self.split_heads(self.W_k(key))    # (B, h, T_k, d_k)
        V = self.split_heads(self.W_v(value))  # (B, h, T_v, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, T_q, T_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (B, h, T_q, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T_q, h, d_k)
        attn_output = attn_output.view(B, T_q, self.d_model)    # (B, T_q, d_model)
        
        return self.W_o(attn_output), attn_weights


class PositionwiseFeedForward(nn.Module):
    """FFN: two linear layers with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # d_ff is typically 4 * d_model (e.g., 512 → 2048)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # GELU activation (used in BERT, GPT)
        # Original transformer used ReLU, but GELU is now standard
    
    def forward(self, x):
        # x: (B, T, d_model)
        x = F.gelu(self.linear1(x))   # (B, T, d_ff)
        x = self.dropout(x)
        x = self.linear2(x)            # (B, T, d_model)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer with Pre-LN (more stable than Post-LN)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LN: normalize before attention (more stable)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)   # Residual connection
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)    # Residual connection
        return x


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention + cross-attention."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)    # Masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)   # Enc-Dec
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (look at past tokens only)
        self_attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        # Cross-attention (query=decoder, key/value=encoder)
        cross_attn_out, _ = self.cross_attn(self.norm2(x), encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_out)
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_causal_mask(self, size: int) -> torch.Tensor:
        """Causal (autoregressive) mask: lower triangular."""
        return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    
    def make_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Padding mask: 1 where real token, 0 where padding."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def encode(self, src, src_mask=None):
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.final_norm(x)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        encoder_output = self.encode(src, src_mask)
        
        # Decoder with causal mask
        if tgt_mask is None:
            tgt_mask = self.make_causal_mask(tgt.size(1)).to(tgt.device)
        
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

# Example usage
# model = Transformer(src_vocab_size=32000, tgt_vocab_size=32000)
# src = torch.randint(0, 32000, (2, 50))  # Batch=2, seq_len=50
# tgt = torch.randint(0, 32000, (2, 45))
# logits = model(src, tgt)  # (2, 45, 32000)
```

### 7.4 Layer Normalization

\[
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

where \( \mu = \frac{1}{d}\sum_i x_i \), \( \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2 \), and \( \gamma, \beta \) are learned parameters.

Unlike BatchNorm (which normalizes across batch), LayerNorm normalizes across the feature dimension — crucial for variable-length sequences.

### 7.5 Training the Transformer

**Label smoothing:** Instead of hard labels (0/1), use soft targets: \( \hat{y}_k = \epsilon/K \) for wrong classes, \( 1 - \epsilon + \epsilon/K \) for the correct class.

**Noam learning rate schedule:**

\[
\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5})
\]

```python
import torch.optim as optim

class NoamScheduler:
    """Learning rate scheduler from 'Attention is All You Need'."""
    
    def __init__(self, d_model: int, warmup_steps: int, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model**(-0.5) * min(
            self.step_num**(-0.5),
            self.step_num * self.warmup_steps**(-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

---

## 8. Tokenizers In Depth

Modern NLP uses subword tokenization — balancing vocabulary size with coverage.

### 8.1 Byte-Pair Encoding (BPE)

BPE (Sennrich et al., 2016) iteratively merges the most frequent byte/character pair.

**Algorithm:**
1. Start with character vocabulary (split all words into characters + `</w>` end symbol)
2. Count all adjacent symbol pairs
3. Merge the most frequent pair → new symbol
4. Repeat for `num_merges` iterations

```python
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class BPETokenizer:
    """Byte-Pair Encoding tokenizer from scratch."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
    
    def _get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """Build initial word frequency dictionary with character splits."""
        word_freq = Counter()
        for text in corpus:
            for word in text.split():
                word_freq[word] += 1
        # Represent each word as space-separated characters + </w>
        vocab = {}
        for word, freq in word_freq.items():
            chars = ' '.join(list(word)) + ' </w>'
            vocab[chars] = freq
        return vocab
    
    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """Count frequency of each symbol pair."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair of symbols in all words."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab
    
    def train(self, corpus: List[str]):
        """Train BPE on corpus."""
        vocab = self._get_vocab(corpus)
        
        # Initial character vocabulary
        self.token_to_id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for word in vocab:
            for char in word.split():
                if char not in self.token_to_id:
                    self.token_to_id[char] = len(self.token_to_id)
        
        # BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            merged_token = ''.join(best_pair)
            self.merges.append(best_pair)
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        print(f"Vocabulary size: {len(self.token_to_id)}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        tokens = []
        for word in text.split():
            chars = list(word) + ['</w>']
            word_tokens = chars
            
            # Apply merges in order
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == pair:
                        new_tokens.append(''.join(pair))
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            tokens.extend(word_tokens)
        return tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(t, 1) for t in tokens]  # 1 = <unk>


# Using HuggingFace tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Train BPE tokenizer on custom corpus
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
# tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Load pre-trained tokenizer
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### 8.2 WordPiece (BERT's Tokenizer)

WordPiece (Schuster & Nakamura, 2012) is similar to BPE but maximizes the likelihood of training data:

\[
\text{score}(A, B) = \frac{\text{freq}(AB)}{\text{freq}(A) \times \text{freq}(B)}
\]

Tokens in the middle of a word are prefixed with `##`: "playing" → `["play", "##ing"]`

### 8.3 SentencePiece (Language-agnostic)

SentencePiece (Kudo & Richardson, 2018) treats the input as a raw byte stream — works for any language without pre-tokenization.

- Uses the input text directly without word boundaries
- Builds vocabulary using BPE or unigram language model
- Represents spaces as a special character `▁` (U+2581)
- "Hello world" → `["▁Hello", "▁world"]`

```python
# SentencePiece
import sentencepiece as spm

# Train
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # >1 for CJK
    model_type='bpe',           # or 'unigram'
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

text = "Hello, this is a test sentence."
print(sp.encode(text, out_type=str))    # Tokens as strings
print(sp.encode(text, out_type=int))    # Token IDs
print(sp.decode(sp.encode(text)))       # Reconstruct
```

### 8.4 Tokenizer Deep Dive with HuggingFace

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Let's test tokenization of the word 'unbelievable'!"
encoding = tokenizer(
    text,
    add_special_tokens=True,   # Add [CLS], [SEP]
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt',       # PyTorch tensors
    return_attention_mask=True,
    return_token_type_ids=True,
    return_offsets_mapping=True  # Character offsets
)

print("Input IDs:", encoding['input_ids'])
print("Attention mask:", encoding['attention_mask'])
print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
# ['[CLS]', 'let', "'", 's', 'test', 'token', '##ization', 'of', 'the', 'word', "'", 
#  'un', '##bel', '##iev', '##able', "'", '!', '[SEP]', ...]

# Batch tokenization
texts = ["First sentence.", "Second, longer sentence here."]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
print("Batch shape:", batch['input_ids'].shape)
```

---

## 9. BERT and Encoder-Only Models

### 9.1 BERT Architecture

BERT (Devlin et al., 2018) is a bidirectional transformer encoder pre-trained with two objectives:

**1. Masked Language Modeling (MLM):**
Randomly mask 15% of tokens. For those:
- 80% → replace with `[MASK]`
- 10% → replace with random token
- 10% → keep unchanged

Predict the original token:
\[
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid \tilde{x})
\]

**2. Next Sentence Prediction (NSP):**
Given sentences A and B (50% real pairs, 50% random):
\[
\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid [CLS] A [SEP] B [SEP])
\]

**Input representation:**
\[
\mathbf{e}_i = \text{TokenEmb}(x_i) + \text{SegmentEmb}(s_i) + \text{PositionEmb}(i)
\]

**BERT-base:** 12 layers, 12 heads, \( d_{\text{model}} = 768 \), \( d_{ff} = 3072 \), ~110M parameters  
**BERT-large:** 24 layers, 16 heads, \( d_{\text{model}} = 1024 \), \( d_{ff} = 4096 \), ~340M parameters

```python
from transformers import (
    BertModel, BertTokenizer, BertForSequenceClassification,
    BertForTokenClassification, BertForQuestionAnswering,
    BertForMaskedLM, AutoModel, AutoTokenizer, Trainer, TrainingArguments
)
import torch
import torch.nn as nn

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get contextual embeddings
text = "The bank can guarantee deposits will eventually cover future tuition costs."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

last_hidden = outputs.last_hidden_state   # (1, T, 768) — token embeddings
pooler_output = outputs.pooler_output     # (1, 768) — [CLS] after linear+tanh

print(f"Token embeddings: {last_hidden.shape}")  # (1, 20, 768)
print(f"[CLS] pooled: {pooler_output.shape}")    # (1, 768)

# All hidden states (each layer's output)
outputs_all = model(**inputs, output_hidden_states=True)
all_hidden = outputs_all.hidden_states   # tuple of 13 tensors (embedding + 12 layers)
print(f"Number of hidden layers: {len(all_hidden)}")  # 13

# Attention patterns
outputs_attn = model(**inputs, output_attentions=True)
attentions = outputs_attn.attentions    # tuple of 12 tensors, each (1, 12, T, T)
```

### 9.2 Fine-tuning BERT for Classification

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label={0: "NEG", 1: "POS"},
    label2id={"NEG": 0, "POS": 1}
)

# Freeze lower layers (optional — speeds up training)
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for i, layer in enumerate(model.bert.encoder.layer):
    if i < 8:  # Freeze first 8 of 12 layers
        for param in layer.parameters():
            param.requires_grad = False

# Training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

training_args = TrainingArguments(
    output_dir='./bert-imdb',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    fp16=True,             # Mixed precision
    dataloader_num_workers=4,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print(results)
```

### 9.3 BERT Variants

| Model | Key Change | Params | Key Benefit |
|-------|-----------|--------|-------------|
| **RoBERTa** | Remove NSP, more data, dynamic masking, larger batches | 125M | Better performance |
| **ALBERT** | Parameter sharing across layers, factorized embedding | 12M | Memory efficient |
| **DistilBERT** | Knowledge distillation, 6 layers | 66M | 40% faster, 97% performance |
| **ELECTRA** | RTD: detect replaced tokens instead of MLM | 110M | More efficient pretraining |
| **DeBERTa** | Disentangled attention (separate pos/content) | 100M-1.5B | State-of-art NLU |
| **XLM-RoBERTa** | Multilingual RoBERTa (100 languages) | 270M | Cross-lingual transfer |

```python
# RoBERTa
from transformers import RobertaTokenizer, RobertaModel
roberta_tok = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')

# ALBERT (very efficient)
from transformers import AlbertTokenizer, AlbertModel
albert_tok = AlbertTokenizer.from_pretrained('albert-base-v2')
albert = AlbertModel.from_pretrained('albert-base-v2')

# DistilBERT (fast inference)
from transformers import DistilBertTokenizer, DistilBertModel
distil_tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil = DistilBertModel.from_pretrained('distilbert-base-uncased')

# ELECTRA (efficient pre-training)
from transformers import ElectraTokenizer, ElectraModel
electra_tok = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
electra = ElectraModel.from_pretrained('google/electra-base-discriminator')
```

---

## 10. GPT and Decoder-Only Models

### 10.1 GPT Architecture

GPT (Generative Pre-trained Transformer) uses only the decoder with causal (left-to-right) self-attention. The objective is **causal language modeling**:

\[
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)
\]

The causal mask ensures token \( t \) only attends to positions \( \leq t \):

\[
M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
\]

**GPT-2 architecture:** 12-48 layers, 12-25 heads, \( d_{\text{model}} \) = 768-1600, trained on WebText (~40GB).

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate with various decoding strategies
def generate_text(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 100,
    strategy: str = 'greedy',
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    num_beams: int = 4,
    num_return_sequences: int = 1
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        if strategy == 'greedy':
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        elif strategy == 'beam':
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                num_return_sequences=num_return_sequences
            )
        elif strategy == 'sampling':
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )
        elif strategy == 'top_k':
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                temperature=temperature
            )
        elif strategy == 'top_p':
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature
            )
    
    return [tokenizer.decode(o, skip_special_tokens=True) for o in output]

# Sampling strategies explained
# Greedy: argmax at each step — deterministic but repetitive
# Beam search: keep top-k sequences — good for translation, not creativity
# Temperature: scale logits before softmax — >1 more random, <1 more peaked
# Top-k: sample from top-k tokens only
# Top-p (nucleus): sample from smallest set of tokens with cumulative prob ≥ p

# Manual top-p sampling
def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative prob above p
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs /= sorted_probs.sum()
    
    # Sample
    next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
    return next_token

# Compute perplexity
def compute_perplexity(model, tokenizer, text: str) -> float:
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return torch.exp(outputs.loss).item()

ppl = compute_perplexity(model, tokenizer, "The quick brown fox jumps over the lazy dog.")
print(f"Perplexity: {ppl:.2f}")
```

---

## 11. Seq2Seq Models: T5 and BART

### 11.1 T5: Text-to-Text Transfer Transformer

T5 (Raffel et al., 2020) frames ALL NLP tasks as text-to-text:

- Translation: `"translate English to German: That is good"` → `"Das ist gut"`
- Summarization: `"summarize: <article>"` → `"<summary>"`
- Classification: `"sst2 sentence: This movie is great"` → `"positive"`
- QA: `"question: What year? context: <passage>"` → `"1984"`

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM

# Load T5
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Summarization
article = """
The Apollo program was the third United States human spaceflight program
carried out by the National Aeronautics and Space Administration (NASA).
It culminated in a series of crewed Moon landing missions.
"""
inputs = t5_tokenizer(
    f"summarize: {article.strip()}",
    max_length=512, truncation=True, return_tensors='pt'
)
summary_ids = t5_model.generate(
    inputs['input_ids'],
    max_new_tokens=100,
    min_length=30,
    num_beams=4,
    early_stopping=True
)
print(t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True))

# Translation
text = "The house is wonderful."
inputs = t5_tokenizer(
    f"translate English to German: {text}",
    return_tensors='pt'
)
output_ids = t5_model.generate(inputs['input_ids'], max_new_tokens=50)
print(t5_tokenizer.decode(output_ids[0], skip_special_tokens=True))
# Das Haus ist wunderbar.

# Use flan-t5 for instruction following
flan_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

prompt = "Classify the sentiment of this review: 'The food was absolutely terrible and cold.'"
inputs = flan_tokenizer(prompt, return_tensors='pt')
outputs = flan_t5.generate(**inputs, max_new_tokens=20)
print(flan_tokenizer.decode(outputs[0], skip_special_tokens=True))  # negative
```

### 11.2 BART: Denoising Seq2Seq

BART (Lewis et al., 2019) pre-trains an encoder-decoder with noising objectives:
- Token masking, deletion, infilling, sentence permutation, document rotation

BART is especially good for: summarization, abstractive QA, data-to-text

```python
from transformers import BartForConditionalGeneration, BartTokenizer

bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tok = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

long_text = """
NASA's Perseverance rover has successfully landed on Mars, marking a historic moment
in space exploration. The rover, which is the size of a small car, touched down in
Jezero Crater — an ancient lakebed — on February 18, 2021. Perseverance carries seven
scientific instruments and will search for signs of ancient microbial life, collect rock
and soil samples, and test oxygen production from the Martian atmosphere.
"""

inputs = bart_tok(long_text, max_length=1024, truncation=True, return_tensors='pt')
summary = bart.generate(
    inputs['input_ids'],
    max_length=130,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
print(bart_tok.decode(summary[0], skip_special_tokens=True))
```

---

## 12. Sentence Transformers and SBERT

### 12.1 The Problem with BERT for Sentence Similarity

Computing cosine similarity between 10,000 sentences using BERT requires \( \binom{10000}{2} = 49,995,000 \) BERT passes — prohibitively slow.

**SBERT** (Sentence-BERT, Reimers & Gurevych, 2019) uses siamese/triplet networks to produce fixed-size sentence embeddings that can be compared with cosine similarity efficiently.

### 12.2 SBERT Training

**Classification objective (NLI):**

\[
o = \text{softmax}(W_t (\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|))
\]

**Regression objective:**

\[
\mathcal{L} = \text{MSE}(\text{cos\_sim}(\mathbf{u}, \mathbf{v}),\ y)
\]

**Triplet objective:**

\[
\mathcal{L} = \max(||\mathbf{s}_a - \mathbf{s}_p||_2 - ||\mathbf{s}_a - \mathbf{s}_n||_2 + \epsilon,\ 0)
\]

```python
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import losses
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import torch

# Load pre-trained SBERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast
sbert_large = SentenceTransformer('all-mpnet-base-v2')  # 768-dim, better

# Encode sentences
sentences = [
    "Machine learning is a branch of artificial intelligence.",
    "AI encompasses various techniques including ML and deep learning.",
    "The cat sat on the mat.",
    "Python is a great programming language.",
    "Deep learning uses artificial neural networks."
]

embeddings = sbert.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
print(f"Embeddings shape: {embeddings.shape}")  # (5, 384)

# Semantic similarity
cos_sim = util.cos_sim(embeddings, embeddings)
print("Cosine similarity matrix:")
print(torch.round(cos_sim, decimals=3))

# Semantic search
query = "What is machine learning?"
query_embedding = sbert.encode(query, convert_to_tensor=True)
scores = util.cos_sim(query_embedding, embeddings)[0]
top = torch.topk(scores, k=3)
print(f"\nTop 3 similar to '{query}':")
for score, idx in zip(top.values, top.indices):
    print(f"  {score:.4f}: {sentences[idx]}")

# Semantic textual similarity (STS)
pairs = [
    ("I like cats.", "I love dogs."),
    ("The flight was delayed.", "The plane was late."),
    ("Python programming", "Machine learning"),
]
e1 = sbert.encode([p[0] for p in pairs], convert_to_tensor=True)
e2 = sbert.encode([p[1] for p in pairs], convert_to_tensor=True)
sim = util.cos_sim(e1, e2)
for i, (p1, p2) in enumerate(pairs):
    print(f"'{p1}' ↔ '{p2}': {sim[i][i]:.4f}")

# Fine-tune SBERT on custom data
train_examples = [
    InputExample(texts=["sentence A", "sentence B"], label=0.9),  # Similar
    InputExample(texts=["machine learning", "deep learning"], label=0.8),
    InputExample(texts=["cat", "automobile"], label=0.1),
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(sbert)

# Train
sbert.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path='./fine-tuned-sbert'
)
```

---

## 13. NLP Tasks

### 13.1 Named Entity Recognition (NER)

NER identifies entities like persons, organizations, and locations in text.

**Common tagging schemes:**
- IOB2: B-ORG, I-ORG, O (Outside)
- BIOES: B (Begin), I (Inside), O, E (End), S (Single)

```python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch

# Pipeline approach
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"
)

text = "Elon Musk founded SpaceX in Hawthorne, California in 2002. He also leads Tesla."
entities = ner_pipeline(text)
for entity in entities:
    print(f"  [{entity['entity_group']}] {entity['word']}: {entity['score']:.4f}")

# Custom NER model
from transformers import AutoModelForTokenClassification

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

ner_model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Training loop with token-level labels
def align_labels_with_tokens(labels, word_ids):
    """Align word-level NER labels with wordpiece tokens."""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)  # Special tokens
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # Continuation of word — use I- tag if B- tag
            label = labels[word_id]
            if label % 2 == 1:  # B- tag
                label += 1      # Convert to I- tag
            new_labels.append(label)
    return new_labels
```

### 13.2 Part-of-Speech Tagging

```python
from transformers import pipeline

pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
text = "The quick brown fox jumps over the lazy dog."
pos_tags = pos_pipeline(text)
for token in pos_tags:
    print(f"  {token['word']:<15} {token['entity']}")

# Using spaCy (faster for production)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
    print(f"  {token.text:<15} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")
```

### 13.3 Text Classification: Sentiment Analysis

```python
from transformers import pipeline, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

# Zero-shot classification (no fine-tuning needed)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "This new AI model is revolutionizing natural language processing."
result = zero_shot(
    text,
    candidate_labels=["technology", "sports", "politics", "entertainment"],
    multi_label=False
)
print(f"Label: {result['labels'][0]}, Score: {result['scores'][0]:.4f}")

# Sentiment analysis
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
texts = ["I absolutely love this!", "This is the worst experience ever.", "It's okay, nothing special."]
for text, result in zip(texts, sentiment(texts)):
    print(f"'{text}' → {result['label']} ({result['score']:.4f})")

# Custom 5-star rating classifier
rating_model = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None
)
review = "The product was decent but the delivery was slow."
ratings = rating_model(review)
print({r['label']: round(r['score'], 3) for r in ratings[0]})
```

---

## 14. Machine Translation and Summarization

### 14.1 Machine Translation

```python
from transformers import MarianMTModel, MarianTokenizer

# Helsinki-NLP models for many language pairs
def translate(text: str, src_lang: str = "en", tgt_lang: str = "de") -> str:
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    inputs = tok(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs, num_beams=4, early_stopping=True)
    return tok.decode(translated[0], skip_special_tokens=True)

print(translate("Hello, how are you?", "en", "fr"))  # Bonjour, comment allez-vous?
print(translate("The future is bright.", "en", "es"))  # El futuro es brillante.

# BLEU score evaluation
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

hypothesis = "It is a guide to action which ensures that the military always obeys the commands".split()
references = [
    "It is a guide to action that ensures that the military will forever heed Party commands".split(),
    "It is the guiding principle which guarantees the military forces always being under the command".split()
]

smoother = SmoothingFunction().method1
bleu1 = sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoother)
bleu4 = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")
```

### 14.2 Summarization

```python
from transformers import pipeline
import evaluate

# Abstractive summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

article = """
Climate change refers to long-term shifts in temperatures and weather patterns.
These shifts may be natural, but since the 1800s, human activities have been
the main driver of climate change, primarily due to the burning of fossil fuels
like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions
that act like a blanket wrapped around the Earth, trapping the sun's heat and
raising temperatures.
"""

summary = summarizer(
    article,
    max_length=100,
    min_length=30,
    do_sample=False,
    num_beams=4,
    length_penalty=1.0
)
print("Summary:", summary[0]['summary_text'])

# ROUGE evaluation
rouge = evaluate.load("rouge")
predictions = ["The machine learning model achieved high accuracy on the test set."]
references = ["The ML model reached excellent performance on testing data."]

results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {results['rouge1']:.4f}")
print(f"ROUGE-2: {results['rouge2']:.4f}")
print(f"ROUGE-L: {results['rougeL']:.4f}")
```

---

## 15. Question Answering

### 15.1 Extractive QA

Extractive QA finds the answer span within a given context.

**SQuAD format:** Predict start and end token positions:

\[
P(\text{start} = i) = \text{softmax}(\mathbf{h}_i^\top \mathbf{w}_s)_i
\]

\[
P(\text{end} = j) = \text{softmax}(\mathbf{h}_j^\top \mathbf{w}_e)_j
\]

```python
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Extractive QA
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
The transformer architecture was introduced in 2017 by Google Brain in the paper
"Attention Is All You Need". It uses multi-head self-attention mechanisms to process
sequences in parallel, unlike RNNs which process sequentially. BERT, GPT, and T5
are all based on the transformer architecture.
"""

questions = [
    "When was the transformer introduced?",
    "Who introduced the transformer?",
    "What is the key mechanism of transformers?"
]

for q in questions:
    result = qa_pipeline(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {result['answer']} (score: {result['score']:.4f})")
    print()

# Generative QA (open-domain)
gen_qa = pipeline("text2text-generation", model="google/flan-t5-base")
answer = gen_qa(f"Question: What is the capital of France? Answer:")
print(answer[0]['generated_text'])

# Long-document QA with sliding window
def qa_long_doc(question: str, long_context: str, model, tokenizer, max_length: int = 512, stride: int = 128):
    """Handle long documents with sliding window approach."""
    # Tokenize with stride for long contexts
    encodings = tokenizer(
        question, long_context,
        max_length=max_length,
        stride=stride,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    all_start_logits = []
    all_end_logits = []
    
    for i in range(encodings['input_ids'].shape[0]):
        chunk = {k: v[i:i+1] for k, v in encodings.items()
                 if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**chunk)
        all_start_logits.append(outputs.start_logits[0])
        all_end_logits.append(outputs.end_logits[0])
    
    # Find best answer across chunks
    # (simplified - production code would handle offsets properly)
    return all_start_logits, all_end_logits
```

---

## 16. Modern Positional Encodings

### 16.1 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) encodes position by rotating query and key vectors. Instead of adding position to embeddings, it multiplies by a rotation matrix:

\[
\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \cdot e^{im\theta}
\]

In 2D: Rotate by angle \( m\theta \):

\[
\begin{pmatrix} q_1' \\ q_2' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}
\]

The attention score between positions \( m \) and \( n \):

\[
\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = f(\mathbf{q}, \mathbf{k}, m-n)
\]

This encodes **relative position** through the **difference** \( m - n \). Used in LLaMA, Mistral, GPT-NeoX.

```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        # theta_i = base^(-2i/dim) for i = 0, ..., dim/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
        # Precompute cos, sin for all positions
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (T, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))
    
    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, :],  # (1, 1, T, dim)
            self.sin_cached[:, :, :seq_len, :]
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last half of features."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply RoPE to query and key."""
    cos = cos.squeeze(1).squeeze(0)  # (T, dim)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, dim)
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 16.2 ALiBi: Attention with Linear Biases

ALiBi (Press et al., 2021) adds a linear bias to attention scores based on distance:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + m \cdot \begin{pmatrix} 0 & -1 & -2 & \cdots \\ 0 & 0 & -1 & \cdots \\ \vdots & & \ddots & \end{pmatrix}\right) V
\]

where \( m \) is a head-specific slope. This provides strong length extrapolation (train on 1024, infer on 4096+).

```python
import torch
import math

def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each attention head."""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    
    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base_slopes = get_slopes_power_of_2(closest_power_of_2)
    extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
    return torch.tensor(base_slopes + extra_slopes[:num_heads - closest_power_of_2])

def build_alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor:
    slopes = get_alibi_slopes(num_heads)         # (H,)
    # Position distance matrix: -|i - j| for i >= j (causal)
    positions = torch.arange(seq_len)
    distances = -(positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (T, T)
    distances = torch.tril(distances)
    bias = slopes.unsqueeze(1).unsqueeze(1) * distances.unsqueeze(0)  # (H, T, T)
    return bias  # Add to attention logits
```

---

## 17. Advanced Techniques

### 17.1 Flash Attention

Flash Attention (Dao et al., 2022) is an IO-aware exact attention that reduces memory from \( O(N^2) \) to \( O(N) \) by tiling the computation and keeping softmax running statistics on-chip:

**Key idea:** Instead of materializing the full \( N \times N \) attention matrix, process in tiles of size \( B_r \times B_c \) that fit in SRAM (fast cache), using the online softmax trick.

```python
# Using Flash Attention via xformers or PyTorch 2.0
import torch
import torch.nn.functional as F

# PyTorch 2.0+ has built-in Flash Attention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,    # Flash attention
    enable_math=False,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.1,
        is_causal=True   # Causal mask
    )

# HuggingFace integration
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
)
```

### 17.2 Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) used in LLaMA-2/3, Mistral: Groups of queries share a single K/V head, reducing memory bandwidth:

- **MHA**: Each query head has its own K/V head — `H` K/V heads
- **MQA**: All query heads share 1 K/V head — memory efficient but lower quality
- **GQA**: `G` groups, each with `H/G` query heads sharing 1 K/V head — best of both

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K and V for each group
        K = K.repeat_interleave(self.num_groups, dim=1)  # (B, H_q, T, d)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # Standard scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H_q, T, d)
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
```

### 17.3 Sparse Attention (Long-Context NLP)

Full self-attention has \(O(n^2)\) complexity in sequence length. **Sparse attention** reduces this by attending to only a subset of positions. Key variants:

**Sliding window (local attention):** Each token attends only to neighbors within window size \(w\):
\[
\text{Attend}(i) = \{j : |i - j| \leq w\}
\]
Used in Longformer (Beltagy et al., 2020): \(O(n \cdot w)\) complexity.

**Strided/block attention:** Attend to every \(k\)-th token plus local block. Used in Sparse Transformer (Child et al., 2019).

**Dilated attention:** Attend to positions at exponentially increasing strides: \(i, i+d, i+2d, \ldots\). Captures long-range dependencies with fewer edges.

```python
def create_sparse_attention_mask(seq_len: int, window_size: int = 512, global_indices: List[int] = None) -> torch.Tensor:
    """Create Longformer-style local + global attention mask.
    
    - Local: each position attends to positions within window_size
    - Global: special tokens (e.g., [CLS]) attend to all and are attended by all
    """
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1
        if global_indices:
            for g in global_indices:
                mask[i, g] = 1
                mask[g, i] = 1
    return mask
```

**When to use:** Document understanding (4K–128K tokens), code completion with long context, retrieval-augmented QA.

### 17.4 Linear Attention (Efficient Alternatives)

**Linear attention** reformulates softmax attention to achieve \(O(n)\) complexity by changing the order of operations:

Standard: \(\text{softmax}(QK^\top)V\) — materializes \(n \times n\) matrix.  
Linear (Katharopoulos et al., 2020): \(\phi(Q) (\phi(K)^\top V)\) — associativity: \((\phi(Q) \phi(K)^\top) V = \phi(Q) (\phi(K)^\top V)\).

Using \(\phi(x) = \text{elu}(x) + 1\) (positive kernel) or \(\phi(x) = \text{softmax}(x)\) per query:

\[
\text{LinearAttn}(Q, K, V) = \frac{\phi(Q) (\sum_j \phi(K_j)^\top V_j)}{\phi(Q) \sum_j \phi(K_j)^\top}
\]

The denominator normalizes; the numerator can be computed left-to-right in \(O(n)\) for causal generation.

```python
def linear_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Linear attention: O(n) instead of O(n^2).
    Q, K, V: (B, T, H, d)
    Uses elu+1 feature map for positive kernel (permutation of softmax)."""
    # Feature map: ELU(x) + 1 ensures non-negativity
    def feature_map(x):
        return F.elu(x) + 1
    
    Q = feature_map(Q)
    K = feature_map(K)
    
    # KV: (B, T, H, d) -- cache for incremental decoding
    KV = torch.einsum('bthd,bthn->bhdn', K, V)  # (B, H, d, d)
    K_sum = K.sum(dim=1)  # (B, H, d)
    
    # Output: Q @ (K^T @ V) / (Q @ K_sum)
    QKV = torch.einsum('bthd,bhdn->bthn', Q, KV)
    QK_sum = torch.einsum('bthd,bhd->bth', Q, K_sum)
    return QKV / (QK_sum.unsqueeze(-1) + eps)
```

**Trade-offs:** Linear attention is fast and memory-efficient but may underperform softmax on tasks requiring sharp, selective attention. Best for: long sequences, edge deployment, real-time applications.

### 17.5 Pitfalls and Common Mistakes

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Attention dropout in wrong place** | Dropping attention weights *after* softmax can zero out entire heads | Use `attn_dropout` on weights before weighted sum, or use `hidden_dropout` on output |
| **Wrong causal mask shape** | Causal mask must be `(1, 1, T_q, T_k)` for broadcasting | Use `mask.unsqueeze(0).unsqueeze(0)` when needed |
| **Forgetting to scale by √d_k** | Unscaled dot products saturate softmax → vanishing gradients | Always divide by \(\sqrt{d_k}\) |
| **Position encoding length mismatch** | Sinusoidal PE has fixed `max_len`; RoPE needs position IDs | Precompute for max expected length; for variable length, interpolate or use ALiBi |
| **Over-tokenizing** | Aggressive subword splitting inflates sequence length and cost | Use appropriate vocab size; avoid unnecessary special tokens |
| **Stopword removal for transformers** | Removing stopwords destroys grammatical structure | **Never** remove stopwords for BERT/GPT — they need full context |
| **Label misalignment in token classification** | Subword tokens share one label; misalignment causes wrong loss | Use `word_ids` from tokenizer; align labels with `-100` for sub-tokens |
| **Using pooling wrong** | `[CLS]` for BERT is pre-trained for NSP; for other tasks, mean pooling may be better | Match pooling to pre-training objective; consider last-layer mean for sentence similarity |
| **Padding side inconsistency** | Left vs right padding affects position IDs and causal masks | Use `padding_side='left'` for decoder-only generation; `'right'` for BERT-style |

**Debugging attention:** Visualize attention patterns to catch anomalies (e.g., diagonal-only = no learning, uniform = collapsed). Use `output_attentions=True` and plot heatmaps.

### 17.6 Complete HuggingFace Pipelines Reference

```python
from transformers import pipeline

# All main pipelines
tasks = {
    "text-classification":      "distilbert-base-uncased-finetuned-sst-2-english",
    "token-classification":     "dbmdz/bert-large-cased-finetuned-conll03-english",
    "question-answering":       "deepset/roberta-base-squad2",
    "summarization":            "facebook/bart-large-cnn",
    "translation":              "Helsinki-NLP/opus-mt-en-fr",
    "text-generation":          "gpt2",
    "fill-mask":                "bert-base-uncased",
    "zero-shot-classification": "facebook/bart-large-mnli",
    "feature-extraction":       "bert-base-uncased",
}

# Feature extraction (get embeddings)
feat_extractor = pipeline("feature-extraction", model="bert-base-uncased", return_tensors=True)
embeddings = feat_extractor("Hello world")
print(f"Embeddings shape: {embeddings.shape}")  # (1, T, 768)

# Fill mask (BERT MLM)
fill_masker = pipeline("fill-mask", model="bert-base-uncased")
results = fill_masker("The capital of France is [MASK].")
for r in results[:3]:
    print(f"  {r['token_str']}: {r['score']:.4f}")
```

---

## Key References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Word2Vec (Mikolov et al.) | 2013 | Efficient word embeddings via skip-gram/CBOW |
| GloVe (Pennington et al.) | 2014 | Global co-occurrence statistics |
| Attention (Bahdanau et al.) | 2015 | Alignment-based attention for seq2seq |
| FastText (Bojanowski et al.) | 2017 | Subword embeddings for OOV handling |
| Transformer (Vaswani et al.) | 2017 | Self-attention replaces recurrence entirely |
| BERT (Devlin et al.) | 2018 | Bidirectional masked language modeling |
| GPT-2 (Radford et al.) | 2019 | Large-scale causal language modeling |
| RoBERTa (Liu et al.) | 2019 | Improved BERT training recipe |
| Sparse Transformer (Child et al.) | 2019 | Strided/block sparse attention |
| SBERT (Reimers & Gurevych) | 2019 | Efficient sentence embeddings |
| BPE (Sennrich et al.) | 2016 | Subword tokenization for NMT |
| Longformer (Beltagy et al.) | 2020 | Sliding-window + global sparse attention |
| Transformers are RNNs (Katharopoulos et al.) | 2020 | Linear attention with O(n) complexity |
| T5 (Raffel et al.) | 2020 | Unified text-to-text framework |
| BART (Lewis et al.) | 2020 | Denoising seq2seq pre-training |
| RoPE (Su et al.) | 2021 | Rotary positional embedding |
| ALiBi (Press et al.) | 2021 | Linear bias for length extrapolation |
| Flash Attention (Dao et al.) | 2022 | IO-aware exact attention |
| GQA (Ainslie et al.) | 2023 | Grouped query attention |

---

*This guide covers NLP from fundamentals to state-of-the-art architectures. For production use, always check the HuggingFace Model Hub for the latest models on your specific task and language.*
