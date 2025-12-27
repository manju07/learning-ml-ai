# Natural Language Processing (NLP) and Transformers: Complete Guide

## Table of Contents
1. [Introduction to NLP](#introduction)
2. [Text Preprocessing](#text-preprocessing)
3. [Traditional NLP Methods](#traditional-nlp)
4. [Word Embeddings](#word-embeddings)
5. [Recurrent Neural Networks for NLP](#rnn-nlp)
6. [Attention Mechanism](#attention)
7. [Transformers Architecture](#transformers)
8. [BERT and Variants](#bert)
9. [GPT and Language Models](#gpt)
10. [Fine-tuning Pre-trained Models](#fine-tuning)
11. [Practical Examples](#examples)
12. [Advanced Topics](#advanced)

---

## Introduction to NLP {#introduction}

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. Modern NLP is dominated by transformer-based models.

### Key NLP Tasks
- **Text Classification**: Sentiment analysis, spam detection
- **Named Entity Recognition (NER)**: Extract entities (person, location, etc.)
- **Machine Translation**: Translate between languages
- **Question Answering**: Answer questions from context
- **Text Generation**: Generate coherent text
- **Summarization**: Create summaries of long texts
- **Sentiment Analysis**: Determine emotional tone

### Evolution of NLP
1. **Rule-based**: Hand-crafted rules
2. **Statistical**: N-grams, TF-IDF
3. **Neural Networks**: RNNs, LSTMs, GRUs
4. **Transformers**: BERT, GPT, T5 (Current state-of-the-art)

---

## Text Preprocessing {#text-preprocessing}

### Basic Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Comprehensive text preprocessing"""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Example
text = "I'm loving this product! It's amazing!!! Check it out: https://example.com"
cleaned = preprocess_text(text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
```

### Advanced Preprocessing

```python
import spacy

# Load spaCy model
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def advanced_preprocess(text):
    """Advanced preprocessing with spaCy"""
    doc = nlp(text)
    
    # Extract tokens, lemmas, POS tags
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    
    # Named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'entities': entities
    }
```

---

## Traditional NLP Methods {#traditional-nlp}

### Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create BoW representation
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(feature_names)}")
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF representation
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(texts)

# Get feature names
feature_names = tfidf.get_feature_names_out()
```

### N-grams

```python
from nltk.util import ngrams

def extract_ngrams(text, n=2):
    """Extract n-grams from text"""
    tokens = word_tokenize(text.lower())
    return list(ngrams(tokens, n))

# Example
text = "Natural language processing is amazing"
bigrams = extract_ngrams(text, n=2)
trigrams = extract_ngrams(text, n=3)
print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")
```

---

## Word Embeddings {#word-embeddings}

### Word2Vec

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Train Word2Vec model
sentences = [word_tokenize(text.lower()) for text in texts]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector
vector = model.wv['machine']
print(f"Vector for 'machine': {vector[:5]}...")

# Find similar words
similar_words = model.wv.most_similar('learning', topn=5)
print(f"Words similar to 'learning': {similar_words}")

# Load pre-trained Word2Vec
# w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
```

### GloVe

```python
# Load pre-trained GloVe embeddings
# Download from: https://nlp.stanford.edu/projects/glove/

def load_glove_embeddings(file_path, embedding_dim=100):
    """Load GloVe embeddings"""
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# embeddings = load_glove_embeddings('glove.6B.100d.txt')
```

### FastText

```python
from gensim.models import FastText

# Train FastText model
model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)

# FastText can handle out-of-vocabulary words
vector = model.wv['unseenword']
```

---

## Recurrent Neural Networks for NLP {#rnn-nlp}

### LSTM for Text Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=100),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Bidirectional LSTM

```python
# Bidirectional LSTM processes sequence in both directions
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=100),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Attention Mechanism {#attention}

### Self-Attention Implementation

```python
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        attention_output = self.combine_heads(attention_output, batch_size)
        output = self.dense(attention_output)
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def combine_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))
```

---

## Transformers Architecture {#transformers}

### Transformer Encoder Block

```python
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
```

### Positional Encoding

```python
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
```

### Complete Transformer Encoder

```python
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [TransformerEncoderBlock(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
```

---

## BERT and Variants {#bert}

### Using BERT with Transformers Library

```python
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize text
text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors='tf', padding=True, truncation=True)

# Get BERT embeddings
outputs = bert_model(encoded)
embeddings = outputs.last_hidden_state
pooled_output = outputs.pooler_output

print(f"Token embeddings shape: {embeddings.shape}")
print(f"Pooled output shape: {pooled_output.shape}")
```

### BERT for Classification

```python
from transformers import TFBertForSequenceClassification

# Load BERT for classification
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Tokenize inputs
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='tf'
)

# Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    labels,
    epochs=3,
    batch_size=16
)
```

### BERT Variants

```python
# DistilBERT - Smaller, faster
from transformers import DistilBertTokenizer, TFDistilBertModel
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# RoBERTa - Improved BERT
from transformers import RobertaTokenizer, TFRobertaModel
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = TFRobertaModel.from_pretrained('roberta-base')

# ALBERT - Parameter-efficient
from transformers import AlbertTokenizer, TFAlbertModel
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
albert_model = TFAlbertModel.from_pretrained('albert-base-v2')
```

---

## GPT and Language Models {#gpt}

### Using GPT-2

```python
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt = "The future of AI is"
inputs = tokenizer.encode(prompt, return_tensors='tf')

# Generate
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### GPT-3/4 Style Prompting

```python
def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.7):
    """Generate text using language model"""
    inputs = tokenizer.encode(prompt, return_tensors='tf')
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
prompt = "Once upon a time"
story = generate_text(prompt, model, tokenizer)
print(story)
```

---

## Fine-tuning Pre-trained Models {#fine-tuning}

### Fine-tuning BERT for Sentiment Analysis

```python
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import create_optimizer
import tensorflow as tf

# Load model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prepare data
def tokenize_data(texts, labels, max_length=128):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    }, tf.constant(labels)

# Tokenize
train_data, train_labels = tokenize_data(train_texts, train_labels)
val_data, val_labels = tokenize_data(val_texts, val_labels)

# Create optimizer with learning rate schedule
num_train_steps = len(train_texts) // 16 * 3
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps
)

# Compile
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    epochs=3,
    batch_size=16
)
```

### Fine-tuning GPT-2 for Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

dataset = Dataset.from_dict({'text': texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

---

## Practical Examples {#examples}

### Example 1: Sentiment Analysis with BERT

```python
from transformers import pipeline

# Use pre-built pipeline
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Analyze sentiment
results = classifier([
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
])

for result in results:
    print(f"Text: {result['label']}, Score: {result['score']:.4f}")
```

### Example 2: Named Entity Recognition

```python
from transformers import pipeline

# NER pipeline
ner = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy='simple')

text = "Apple is looking at buying U.K. startup for $1 billion"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']}")
```

### Example 3: Question Answering

```python
from transformers import pipeline

# QA pipeline
qa = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

context = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data.
"""

question = "What is machine learning?"
answer = qa(question=question, context=context)
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
print(f"Score: {answer['score']:.4f}")
```

### Example 4: Text Summarization

```python
from transformers import pipeline

# Summarization pipeline
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

text = """
Machine learning is transforming industries across the globe. From healthcare to finance,
companies are leveraging ML to make better decisions. The technology continues to evolve
rapidly, with new architectures and techniques emerging regularly.
"""

summary = summarizer(text, max_length=50, min_length=30, do_sample=False)
print(f"Summary: {summary[0]['summary_text']}")
```

---

## Advanced Topics {#advanced}

### Transfer Learning Strategies

```python
# 1. Feature Extraction (Frozen base)
base_model.trainable = False

# 2. Fine-tuning (Unfreeze some layers)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# 3. Progressive Unfreezing
# Unfreeze layers gradually during training
```

### Model Compression

```python
# Knowledge Distillation
from transformers import DistilBertForSequenceClassification

# Train smaller model to mimic larger model
student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
# Train student to match teacher predictions
```

### Multi-task Learning

```python
# Shared encoder with task-specific heads
shared_encoder = TFBertModel.from_pretrained('bert-base-uncased')

# Task 1: Classification
classifier_head = layers.Dense(num_classes, activation='softmax')

# Task 2: NER
ner_head = layers.Dense(num_ner_tags, activation='softmax')

# Use shared encoder for both tasks
```

---

## Best Practices

1. **Use Pre-trained Models**: Start with BERT, GPT, etc.
2. **Fine-tune Carefully**: Use appropriate learning rates (2e-5 for BERT)
3. **Handle Long Sequences**: Use truncation and chunking
4. **Monitor Training**: Watch for overfitting
5. **Use Appropriate Tokenizers**: Match tokenizer to model
6. **Batch Processing**: Process in batches for efficiency
7. **Gradient Accumulation**: For large models with small batches
8. **Mixed Precision**: Use FP16 for faster training

---

## Resources

- **Hugging Face**: transformers library and models
- **Papers**: 
  - Attention Is All You Need (2017)
  - BERT (2018)
  - GPT-2/3 (2019/2020)
- **Datasets**: GLUE, SQuAD, WikiText
- **Tools**: spaCy, NLTK, Transformers

---

## Conclusion

NLP has been revolutionized by transformers. Key takeaways:

1. **Start with Pre-trained Models**: BERT, GPT, etc.
2. **Understand Attention**: Core of transformer architecture
3. **Fine-tune Appropriately**: Use correct learning rates
4. **Use Transformers Library**: Simplifies implementation
5. **Experiment**: Try different models for your task

Remember: Transformers are powerful but require understanding of attention mechanisms and proper fine-tuning!

