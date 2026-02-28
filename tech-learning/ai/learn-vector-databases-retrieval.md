# Vector Databases & Dense Retrieval: Complete Guide

## Table of Contents
1. [Introduction to Vector Retrieval](#introduction-to-vector-retrieval)
2. [Embeddings](#embeddings)
3. [Similarity Metrics](#similarity-metrics)
4. [Approximate Nearest Neighbor (ANN)](#approximate-nearest-neighbor-ann)
5. [Vector Databases](#vector-databases)
6. [Hybrid Search](#hybrid-search)
7. [Reranking](#reranking)
8. [Chunking Strategies](#chunking-strategies)
9. [Advanced RAG Patterns](#advanced-rag-patterns)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)

---

## Introduction to Vector Retrieval

**Vector retrieval** finds items similar to a query by comparing dense vector representations (embeddings). Core to RAG, semantic search, and recommendation.

### Sparse vs Dense Retrieval

| Type | Representation | Example | Use Case |
|------|-----------------|---------|----------|
| **Sparse** | Bag-of-words, TF-IDF, BM25 | Lexical overlap | Keyword match |
| **Dense** | Neural embeddings | Semantic similarity | Meaning match |

### Why Dense?

- **Semantic**: "car" and "automobile" are close
- **Multilingual**: Cross-language similarity
- **Robust**: Handles paraphrasing, typos

---

## Embeddings

### Embedding Models

| Model | Dimension | Use Case |
|-------|-----------|----------|
| **OpenAI text-embedding-3** | 1536, 3072 | General, API |
| **BAAI/bge-large** | 1024 | Open-source, strong |
| **sentence-transformers** | 384–1024 | Local, many options |
| **Cohere embed** | 1024 | Multilingual |

### Generating Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Single
embedding = model.encode("Machine learning is a subset of AI")
print(embedding.shape)  # (1024,)

# Batch
texts = ["First document", "Second document", "Third document"]
embeddings = model.encode(texts, batch_size=32)
```

### OpenAI Embeddings

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)
embedding = response.data[0].embedding
```

### Embedding Normalization

For cosine similarity, normalize embeddings:

```python
import numpy as np

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)
# Then dot product = cosine similarity
```

---

## Similarity Metrics

### Cosine Similarity

sim(a, b) = (a · b) / (||a|| ||b||)

Range [-1, 1]; 1 = identical direction.

```python
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([query_embedding], doc_embeddings)[0]
```

### Dot Product

Use when embeddings are normalized: dot product = cosine similarity.

### Euclidean Distance

L2 distance; convert to similarity: sim = -distance or 1/(1+distance).

```python
from scipy.spatial.distance import cdist
distances = cdist([query_embedding], doc_embeddings, metric='euclidean')[0]
```

---

## Approximate Nearest Neighbor (ANN)

Exact k-NN is O(n) per query. ANN trades some accuracy for speed.

### HNSW (Hierarchical Navigable Small World)

- Graph-based
- Build time: O(n log n)
- Query: O(log n)
- Good recall/speed tradeoff

### IVF (Inverted File Index)

- Cluster vectors
- Search only in nearest cluster(s)
- Faiss IVF_FLAT, IVF_PQ

### Product Quantization (PQ)

- Compress vectors to codes
- Distance via lookup tables
- Memory efficient

### FAISS

```python
import faiss
import numpy as np

dim = 768
n_vectors = 100000

# Build index
embeddings = np.random.randn(n_vectors, dim).astype('float32')
faiss.normalize_L2(embeddings)  # For cosine

index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.add(embeddings)

# Search
query = np.random.randn(1, dim).astype('float32')
faiss.normalize_L2(query)
D, I = index.search(query, k=10)  # Top 10
```

### ANN Benchmark

- Recall@k: % of true k-NN in returned results
- QPS: Queries per second

---

## Vector Databases

### Chroma

```python
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

# Add
collection.add(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    ids=["id1", "id2", "id3"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}]
)

# Query
results = collection.query(
    query_texts=["Search query"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

### Pinecone

```python
import pinecone
from pinecone import ServerlessSpec

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

# Upsert
index.upsert(vectors=[
    ("id1", [0.1, 0.2, ...], {"meta": "value"}),
    ("id2", [0.3, 0.4, ...], {"meta": "value2"})
])

# Query
results = index.query(vector=query_vector, top_k=10, include_metadata=True)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(path="./qdrant_db")
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

client.upsert(
    collection_name="docs",
    points=[
        PointStruct(id=1, vector=vec1, payload={"text": "..."}),
        PointStruct(id=2, vector=vec2, payload={"text": "..."})
    ]
)

results = client.search(collection_name="docs", query_vector=query_vec, limit=10)
```

### Weaviate, Milvus, pgvector

- **Weaviate**: GraphQL, hybrid search
- **Milvus**: Scalable, distributed
- **pgvector**: PostgreSQL extension, good for existing DBs

```python
# pgvector
# CREATE EXTENSION vector;
# CREATE TABLE docs (id SERIAL, embedding vector(768), content text);
# CREATE INDEX ON docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Hybrid Search

Combine **dense** (semantic) and **sparse** (BM25) for better recall.

### Reciprocal Rank Fusion (RRF)

```python
def rrf(rankings_list, k=60):
    """Fuse multiple rankings with RRF"""
    scores = {}
    for rankings in rankings_list:
        for rank, doc_id in enumerate(rankings):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# dense_ranking = [2, 1, 5, 3, 4]
# sparse_ranking = [1, 3, 2, 5, 4]
# fused = rrf([dense_ranking, sparse_ranking])
```

### LangChain Hybrid

```python
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever

vector_retriever = Chroma.from_documents(docs, embedding).as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
docs = ensemble.get_relevant_documents("query")
```

---

## Reranking

Rerank top-K from first-stage retriever with a more accurate (slower) model.

### Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")
pairs = [[query, doc] for doc in retrieved_docs]
scores = reranker.predict(pairs)
top_indices = np.argsort(scores)[::-1][:5]
reranked_docs = [retrieved_docs[i] for i in top_indices]
```

### ColBERT (Late Interaction)

Token-level interaction; efficient via MaxSim.

### Cohere Rerank API

```python
import cohere
co = cohere.Client("...")
results = co.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=5)
```

---

## Chunking Strategies

### Fixed Size

```python
def chunk_fixed(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
```

### Sentence-Based

```python
from nltk.tokenize import sent_tokenize

def chunk_sentences(text, sentences_per_chunk=5):
    sents = sent_tokenize(text)
    return [" ".join(sents[i:i+sentences_per_chunk]) for i in range(0, len(sents), sentences_per_chunk)]
```

### Semantic Chunking

Split on topic changes using embedding similarity between sentences.

```python
def semantic_chunk(text, threshold=0.7):
    sents = sent_tokenize(text)
    embeddings = model.encode(sents)
    chunks = []
    current = [sents[0]]
    for i in range(1, len(sents)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sents[i])
    if current:
        chunks.append(" ".join(current))
    return chunks
```

### Recursive Character Splitter (LangChain)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(long_text)
```

---

## Advanced RAG Patterns

### Query Expansion

```python
def expand_query(llm, query):
    prompt = f"Generate 2 alternative phrasings of: {query}"
    alternatives = llm.generate(prompt)
    return [query] + parse_alternatives(alternatives)
# Retrieve with each, union results
```

### HyDE (Hypothetical Document Embeddings)

```python
# Generate hypothetical answer with LLM
hypo_doc = llm.generate(f"Write a passage that answers: {query}")
# Embed hypo_doc, retrieve similar (real) docs
```

### Multi-Hop / Iterative

```python
# 1. Retrieve initial docs
# 2. Synthesize sub-query from docs
# 3. Retrieve with sub-query
# 4. Combine and generate
```

### Self-RAG

- Retrieve when uncertain
- Criticize retrieved docs
- Critique own answer

---

## Practical Examples

### Example 1: Full RAG with Chroma + Reranker

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
reranker = CrossEncoder("BAAI/bge-reranker-large")

# Index
chunks = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(chunks, embeddings)

# Retrieve + Rerank
def rag_query(query, k_retrieve=20, k_rerank=5):
    docs = vectorstore.similarity_search(query, k=k_retrieve)
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    top_idx = np.argsort(scores)[::-1][:k_rerank]
    return [docs[i] for i in top_idx]
```

### Example 2: Hybrid with FAISS + BM25

```python
import faiss
from rank_bm25 import BM25Okapi

# Dense
index = faiss.IndexFlatIP(768)
index.add(normalize(embeddings))

# Sparse
tokenized = [d.split() for d in documents]
bm25 = BM25Okapi(tokenized)

def hybrid_search(query, k=10):
    q_emb = model.encode([query])[0]
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1).astype('float32'), k=50)
    dense_ranking = I[0].tolist()

    bm25_scores = bm25.get_scores(query.split())
    sparse_ranking = np.argsort(bm25_scores)[::-1][:50].tolist()

    fused = rrf([dense_ranking, sparse_ranking])[:k]
    return [documents[i] for i in fused]
```

---

## Best Practices

1. **Chunk size**: 256–512 tokens often good; tune per domain
2. **Overlap**: 10–20% overlap reduces boundary issues
3. **Embedding model**: Match to domain (e.g., code, medical)
4. **Rerank**: Use when latency allows
5. **Hybrid**: When keywords matter (names, IDs)
6. **Metadata filtering**: Pre-filter before vector search when possible
7. **Evaluation**: Recall@k, MRR on labeled (query, relevant_doc) pairs

---

## Summary

| Component | Options |
|-----------|---------|
| Embeddings | OpenAI, BGE, sentence-transformers |
| Vector DB | Chroma, Pinecone, Qdrant, pgvector, FAISS |
| ANN | HNSW, IVF, PQ |
| Hybrid | RRF, weighted combination |
| Reranking | Cross-encoder, ColBERT, Cohere |
| Chunking | Fixed, sentence, semantic, recursive |

**Libraries**: `sentence-transformers`, `faiss`, `chromadb`, `langchain`, `qdrant-client`
