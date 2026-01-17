# Large Language Models (LLMs) and RAG: Complete Guide

## Table of Contents
1. [Introduction to LLMs](#introduction-to-llms)
2. [Prompt Engineering](#prompt-engineering)
3. [Fine-tuning LLMs](#fine-tuning-llms)
4. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
5. [RAG Architecture](#rag-architecture)
6. [Vector Databases](#vector-databases)
7. [Embeddings](#embeddings)
8. [Chunking Strategies](#chunking-strategies)
9. [RAG Implementation](#rag-implementation)
10. [Advanced RAG Techniques](#advanced-rag-techniques)
11. [Practical Examples](#practical-examples)
12. [Best Practices](#best-practices)

---

## Introduction to LLMs

Large Language Models (LLMs) are transformer-based models trained on vast amounts of text data. They can understand and generate human-like text.

### Popular LLMs

- **GPT-4**: OpenAI's latest model
- **GPT-3.5**: ChatGPT's model
- **Claude**: Anthropic's model
- **Llama 2/3**: Meta's open-source models
- **Mistral**: Efficient open-source model
- **Gemini**: Google's model

### LLM Capabilities

- **Text Generation**: Creative writing, code generation
- **Question Answering**: Answer questions from context
- **Summarization**: Summarize long texts
- **Translation**: Translate between languages
- **Classification**: Classify text
- **Conversation**: Chat and dialogue

---

## Prompt Engineering

### Basic Prompting

```python
from openai import OpenAI

client = OpenAI()

# Zero-shot prompting
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

# Few-shot prompting
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": """
        Example 1: The sky is blue -> positive
        Example 2: I hate this -> negative
        Example 3: This is amazing -> positive
        Example 4: Terrible service -> ?
        """}
    ]
)
```

### Advanced Prompting Techniques

```python
# Chain-of-Thought (CoT)
prompt = """
Solve this step by step:
If a train travels 60 mph for 2 hours, how far does it go?

Let's think step by step:
1. Speed = 60 mph
2. Time = 2 hours
3. Distance = Speed × Time
4. Distance = 60 × 2 = 120 miles
"""

# Role-based prompting
prompt = """
You are an expert data scientist. Explain machine learning 
to a beginner in simple terms.
"""

# Template prompting
template = """
Task: {task}
Context: {context}
Instructions: {instructions}
Output: 
"""

prompt = template.format(
    task="Summarize the following text",
    context=text,
    instructions="Keep it under 100 words"
)
```

### Prompt Templates

```python
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

# System prompt template
system_template = PromptTemplate(
    "You are a helpful assistant specialized in {domain}. "
    "Always provide accurate and detailed responses."
)

# User prompt template
user_template = PromptTemplate(
    "Question: {question}\n"
    "Context: {context}\n"
    "Please provide a comprehensive answer."
)

# Use templates
system_prompt = system_template.format(domain="machine learning")
user_prompt = user_template.format(
    question="What is overfitting?",
    context="Machine learning model training"
)
```

---

## Fine-tuning LLMs

### Fine-tuning with OpenAI

```python
import openai

# Prepare training data
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."}
        ]
    },
    # More examples...
]

# Create fine-tuning job
response = openai.FineTuningJob.create(
    training_file="training_data.jsonl",
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate_multiplier": 0.1
    }
)

# Check status
job_id = response.id
status = openai.FineTuningJob.retrieve(job_id)
```

### Fine-tuning with Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

### LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # See trainable parameters

# Train (only LoRA weights are updated)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
```

---

## Retrieval Augmented Generation (RAG)

### What is RAG?

RAG combines retrieval (finding relevant information) with generation (creating responses). It allows LLMs to access external knowledge without fine-tuning.

### RAG Benefits

- **Up-to-date Information**: Access current data
- **Domain-specific Knowledge**: Use custom documents
- **Reduced Hallucination**: Grounded in retrieved facts
- **Transparency**: Can cite sources
- **Cost-effective**: No need to fine-tune

---

## RAG Architecture

### Basic RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, llm, vector_store, retriever):
        self.llm = llm
        self.vector_store = vector_store
        self.retriever = retriever
    
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant documents"""
        query_embedding = self.get_embedding(query)
        results = self.vector_store.similarity_search(
            query_embedding, k=top_k
        )
        return results
    
    def generate(self, query: str, context: str):
        """Generate response using retrieved context"""
        prompt = f"""
        Context: {context}
        
        Question: {query}
        
        Answer the question using only the context provided.
        If the answer is not in the context, say "I don't know".
        """
        
        response = self.llm.generate(prompt)
        return response
    
    def query(self, query: str):
        """Complete RAG pipeline"""
        # Retrieve
        documents = self.retrieve(query)
        context = "\n".join([doc.content for doc in documents])
        
        # Generate
        answer = self.generate(query, context)
        
        return {
            "answer": answer,
            "sources": documents
        }
```

---

## Vector Databases

### ChromaDB

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Document 1 text", "Document 2 text"],
    ids=["id1", "id2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# Query
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=5
)
```

### Pinecone

```python
import pinecone

# Initialize
pinecone.init(api_key="your-key", environment="us-east1-gcp")

# Create index
index = pinecone.Index("documents")

# Upsert vectors
vectors = [
    ("id1", embedding1, {"text": "Document 1"}),
    ("id2", embedding2, {"text": "Document 2"})
]
index.upsert(vectors=vectors)

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

### FAISS

```python
import faiss
import numpy as np

# Create index
dimension = 768  # Embedding dimension
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.array([embedding1, embedding2, ...]).astype('float32')
index.add(vectors)

# Search
query_vector = np.array([query_embedding]).astype('float32')
k = 5
distances, indices = index.search(query_vector, k)

# Get results
results = [vectors[i] for i in indices[0]]
```

---

## Embeddings

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

# Get embedding
text = "Machine learning is a subset of AI"
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=text
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text
texts = [
    "Machine learning is fascinating",
    "AI is transforming industries",
    "Deep learning uses neural networks"
]

embeddings = model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")

# Find similar
query = "artificial intelligence"
query_embedding = model.encode([query])

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embedding, embeddings)
most_similar_idx = np.argmax(similarities[0])
print(f"Most similar: {texts[most_similar_idx]}")
```

---

## Chunking Strategies

### Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Recursive character splitter (recommended)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(long_text)

# Character splitter
char_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator=" "
)

chunks = char_splitter.split_text(text)
```

### Semantic Chunking

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)

chunks = semantic_splitter.create_documents([long_text])
```

---

## RAG Implementation

### Complete RAG System

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = DirectoryLoader('./documents', glob="*.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
query = "What is machine learning?"
result = qa_chain({"query": query})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")
```

### Advanced RAG with LangChain

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    retriever=retriever,
    memory=memory
)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == 'quit':
        break
    
    result = qa_chain({"question": query})
    print(f"Assistant: {result['answer']}")
```

---

## Advanced RAG Techniques

### Re-ranking

```python
from sentence_transformers import CrossEncoder

# Cross-encoder for re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, documents, top_k=5):
    """Re-rank retrieved documents"""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]

# Use in RAG pipeline
retrieved_docs = retriever.retrieve(query)
reranked_docs = rerank_results(query, retrieved_docs)
```

### Multi-query RAG

```python
def generate_queries(original_query: str, llm) -> list:
    """Generate multiple query variations"""
    prompt = f"""
    Given the following question, generate 3 different ways to ask the same question.
    Original question: {original_query}
    
    Generate 3 variations:
    1.
    2.
    3.
    """
    
    response = llm.generate(prompt)
    queries = parse_queries(response)
    return queries

# Use multiple queries
queries = generate_queries(original_query, llm)
all_docs = []
for query in queries:
    docs = retriever.retrieve(query)
    all_docs.extend(docs)

# Deduplicate and rerank
unique_docs = list(set(all_docs))
final_docs = rerank_results(original_query, unique_docs)
```

### Parent-Child Chunking

```python
class ParentChildChunker:
    """Chunk with parent-child relationships"""
    def __init__(self, chunk_size=500, parent_chunk_size=2000):
        self.chunk_size = chunk_size
        self.parent_chunk_size = parent_chunk_size
    
    def chunk(self, text):
        # Create parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size
        )
        parent_chunks = parent_splitter.split_text(text)
        
        # Create child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size
        )
        
        chunks = []
        for parent in parent_chunks:
            children = child_splitter.split_text(parent)
            for child in children:
                chunks.append({
                    "content": child,
                    "parent": parent,
                    "metadata": {"parent_id": hash(parent)}
                })
        
        return chunks
```

---

## Practical Examples

### Example 1: Document Q&A System

```python
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split and embed
chunks = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create Q&A system
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
answer = qa_chain.run("What are the main findings?")
```

### Example 2: Code Documentation RAG

```python
# Load code files
code_files = glob.glob("src/**/*.py", recursive=True)
documents = []

for file in code_files:
    with open(file, 'r') as f:
        content = f.read()
        documents.append({
            "content": content,
            "metadata": {"file": file}
        })

# Create RAG system
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Query codebase
answer = qa_chain.run("How does the authentication work?")
```

---

## Best Practices

1. **Chunk Size**: 500-1000 tokens typically works well
2. **Overlap**: 10-20% overlap between chunks
3. **Embeddings**: Use domain-specific embeddings when possible
4. **Re-ranking**: Use re-ranking for better results
5. **Metadata**: Store metadata for filtering
6. **Hybrid Search**: Combine keyword and semantic search
7. **Evaluation**: Test with diverse queries
8. **Monitoring**: Track retrieval quality

---

## Resources

- **Hugging Face**: Models and datasets
- **LangChain**: RAG framework
- **Vector DBs**: ChromaDB, Pinecone, Weaviate
- **Papers**: RAG (2020), In-Context Learning

---

## Conclusion

LLMs and RAG enable powerful AI applications. Key takeaways:

1. **Master Prompting**: Effective prompts are crucial
2. **Use RAG**: For domain-specific knowledge
3. **Choose Right Embeddings**: Match to your domain
4. **Optimize Chunking**: Better chunks = better retrieval
5. **Evaluate**: Test with real queries

Remember: RAG makes LLMs more accurate and useful for specific domains!

