"""
Text Classification with LSTM
Demonstrates deep learning for NLP tasks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Example: Sentiment analysis dataset
# In practice, load your own dataset
texts = [
    "I love this product! It's amazing.",
    "This is terrible. Worst purchase ever.",
    "Great service and fast delivery.",
    "Not worth the money. Very disappointed.",
    "Excellent quality. Highly recommend!",
    # Add more samples...
]

labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

# For demonstration, create synthetic data
def create_synthetic_data(n_samples=1000):
    """Create synthetic text classification data"""
    positive_texts = [
        f"Great product! I really love it. Highly recommend!",
        f"Excellent quality and fast shipping. Very satisfied!",
        f"Amazing service! Best purchase I've made.",
        f"Outstanding product. Exceeded my expectations!",
        f"Perfect! Exactly what I was looking for."
    ]
    
    negative_texts = [
        f"Terrible product. Very disappointed with quality.",
        f"Poor service. Not worth the money at all.",
        f"Worst purchase ever. Complete waste of money.",
        f"Low quality. Broke after just one use.",
        f"Not satisfied. Would not recommend to anyone."
    ]
    
    texts = []
    labels = []
    
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(positive_texts))
        labels.append(1)
        texts.append(np.random.choice(negative_texts))
        labels.append(0)
    
    return texts, np.array(labels)

# Create synthetic data
texts, labels = create_synthetic_data(n_samples=1000)

# Split data
texts_train, texts_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenize texts
vocab_size = 10000
max_length = 100
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(texts_train)

# Convert texts to sequences
X_train = tokenizer.texts_to_sequences(texts_train)
X_test = tokenizer.texts_to_sequences(texts_test)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Sequence length: {max_length}")

# Build LSTM model
model = keras.Sequential([
    # Embedding layer
    layers.Embedding(vocab_size, 128, input_length=max_length),
    
    # LSTM layers
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    
    # Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lstm_training_history.png')
print("\nTraining history plot saved as 'lstm_training_history.png'")

# Test on new text
def predict_sentiment(text):
    """Predict sentiment for a new text"""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return sentiment, confidence

# Example predictions
test_texts = [
    "I absolutely love this! Best product ever!",
    "This is terrible. Very disappointed.",
    "It's okay, nothing special."
]

print("\nExample Predictions:")
for text in test_texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

