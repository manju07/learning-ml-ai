# Deep Learning: Complete Guide with Examples

## Table of Contents
1. [Introduction to Deep Learning](#introduction)
2. [Neural Networks Fundamentals](#neural-networks)
3. [Activation Functions](#activation-functions)
4. [Backpropagation](#backpropagation)
5. [Optimization Algorithms](#optimization)
6. [Regularization Techniques](#regularization)
7. [Architectures](#architectures)
8. [Transfer Learning](#transfer-learning)
9. [Practical Examples](#examples)
10. [Tools and Frameworks](#tools)

---

## Introduction to Deep Learning {#introduction}

Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to learn hierarchical representations of data. Unlike traditional ML, deep learning can automatically discover features from raw data.

### Key Characteristics
- **Hierarchical Feature Learning**: Each layer learns increasingly complex features
- **Automatic Feature Extraction**: No manual feature engineering required
- **Scalability**: Performance improves with more data
- **Versatility**: Works with images, text, audio, and structured data

### When to Use Deep Learning
- ✅ Large amounts of data available
- ✅ Complex patterns in data
- ✅ Non-linear relationships
- ✅ Image, text, or sequence data
- ❌ Small datasets (< 1000 samples)
- ❌ Need for interpretability
- ❌ Limited computational resources

---

## Neural Networks Fundamentals {#neural-networks}

### Perceptron: The Building Block

A perceptron is the simplest neural network unit:

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            error_count = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Update rule
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
                if y[idx] != y_predicted:
                    error_count += 1
            
            self.errors.append(error_count)
            if error_count == 0:
                break
    
    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron.fit(X, y)
predictions = perceptron.predict(X)
print(f"Predictions: {predictions}")
```

### Multi-Layer Perceptron (MLP)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate non-linearly separable data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Build MLP
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate
test_loss, test_accuracy = model.evaluate(X, y, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

---

## Activation Functions {#activation-functions}

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

### Common Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# ELU
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Swish
def swish(x):
    return x * sigmoid(x)

# Plot all functions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
functions = [
    (sigmoid, 'Sigmoid'),
    (tanh, 'Tanh'),
    (relu, 'ReLU'),
    (leaky_relu, 'Leaky ReLU'),
    (elu, 'ELU'),
    (swish, 'Swish')
]

for idx, (func, name) in enumerate(functions):
    ax = axes[idx // 3, idx % 3]
    ax.plot(x, func(x))
    ax.set_title(name)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

### When to Use Which Activation

- **Sigmoid**: Output layer for binary classification (0-1 probability)
- **Tanh**: Hidden layers, output range (-1, 1)
- **ReLU**: Most common for hidden layers, fast convergence
- **Leaky ReLU**: Prevents dying ReLU problem
- **Softmax**: Output layer for multi-class classification

### Example: Comparing Activations

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Test different activations
activations = ['relu', 'tanh', 'sigmoid', 'elu']
results = {}

for activation in activations:
    model = Sequential([
        Dense(64, activation=activation, input_shape=(784,)),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on MNIST (example)
    # history = model.fit(X_train, y_train, epochs=5, verbose=0)
    # results[activation] = history.history['accuracy'][-1]

print("Activation Function Comparison:")
for act, acc in results.items():
    print(f"{act}: {acc:.4f}")
```

---

## Backpropagation {#backpropagation}

Backpropagation is the algorithm used to train neural networks by computing gradients and updating weights.

### Manual Backpropagation Example

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]  # Number of samples
        
        # Output layer error
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn.train(X, y, epochs=1000)

# Test
predictions = nn.forward(X)
print("\nPredictions:")
print(predictions)
```

---

## Optimization Algorithms {#optimization}

### Gradient Descent Variants

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample loss landscape
def loss_function(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2

# 1. Batch Gradient Descent
def batch_gradient_descent(x_init, learning_rate=0.1, epochs=100):
    x = x_init
    history = [x]
    
    for _ in range(epochs):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x)
    
    return x, history

# 2. Stochastic Gradient Descent (SGD)
def sgd(x_init, learning_rate=0.1, epochs=100):
    x = x_init
    history = [x]
    
    for _ in range(epochs):
        # In real scenario, use random sample
        grad = gradient(x) + np.random.normal(0, 0.1)  # Add noise
        x = x - learning_rate * grad
        history.append(x)
    
    return x, history

# 3. Momentum
def momentum(x_init, learning_rate=0.1, momentum_coef=0.9, epochs=100):
    x = x_init
    v = 0  # velocity
    history = [x]
    
    for _ in range(epochs):
        grad = gradient(x)
        v = momentum_coef * v - learning_rate * grad
        x = x + v
        history.append(x)
    
    return x, history

# 4. Adam Optimizer (simplified)
def adam(x_init, learning_rate=0.01, epochs=100):
    x = x_init
    m = 0  # First moment
    v = 0  # Second moment
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    history = [x]
    
    for t in range(1, epochs + 1):
        grad = gradient(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x)
    
    return x, history

# Compare optimizers
x_init = 5.0
x_batch, hist_batch = batch_gradient_descent(x_init)
x_sgd, hist_sgd = sgd(x_init)
x_momentum, hist_momentum = momentum(x_init)
x_adam, hist_adam = adam(x_init)

plt.figure(figsize=(12, 6))
plt.plot(hist_batch, label='Batch GD', alpha=0.7)
plt.plot(hist_sgd, label='SGD', alpha=0.7)
plt.plot(hist_momentum, label='Momentum', alpha=0.7)
plt.plot(hist_adam, label='Adam', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

### Using Optimizers in TensorFlow/Keras

```python
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad

# SGD with momentum
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Adam optimizer
adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# RMSprop
rmsprop_optimizer = RMSprop(learning_rate=0.001, rho=0.9)

# Adagrad
adagrad_optimizer = Adagrad(learning_rate=0.01)

# Use in model
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Regularization Techniques {#regularization}

### 1. Dropout

```python
from tensorflow.keras import layers, Sequential

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.5),  # Drop 50% of neurons randomly
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Drop 30% of neurons
    layers.Dense(10, activation='softmax')
])
```

### 2. L1 and L2 Regularization

```python
from tensorflow.keras import regularizers

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,),
                 kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),  # L1+L2
    layers.Dense(10, activation='softmax')
])
```

### 3. Batch Normalization

```python
model = Sequential([
    layers.Dense(128, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])
```

### 4. Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stopping]
)
```

### 5. Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

---

## Architectures {#architectures}

### 1. Feedforward Neural Network

```python
def create_feedforward_nn(input_dim, hidden_dims, output_dim, dropout_rate=0.5):
    model = Sequential()
    
    # Input layer
    model.add(layers.Dense(hidden_dims[0], activation='relu', input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Hidden layers
    for dim in hidden_dims[1:]:
        model.add(layers.Dense(dim, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(output_dim, activation='softmax'))
    
    return model

# Example
model = create_feedforward_nn(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10
)
model.summary()
```

### 2. Autoencoder

```python
def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    encoder = Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(encoding_dim, activation='relu')
    ])
    
    # Decoder
    decoder = Sequential([
        layers.Dense(64, activation='relu', input_shape=(encoding_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = Sequential([encoder, decoder])
    
    return autoencoder, encoder, decoder

autoencoder, encoder, decoder = create_autoencoder(input_dim=784, encoding_dim=32)

autoencoder.compile(optimizer='adam', loss='mse')
```

### 3. Variational Autoencoder (VAE)

```python
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim * 2)  # Mean and log variance
        ])
        
        # Decoder
        self.decoder = Sequential([
            layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='sigmoid')
        ])
    
    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def encode(self, x):
        z = self.encoder(x)
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

# Loss function for VAE
def vae_loss(x, reconstructed, z_mean, z_log_var):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(x, reconstructed)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    
    return reconstruction_loss + kl_loss
```

---

## Transfer Learning {#transfer-learning}

### Using Pre-trained Models

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras import layers, Model

# Load pre-trained model (without top layers)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classifier
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Fine-tuning

```python
# Unfreeze some layers for fine-tuning
base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Practical Examples {#examples}

### Example 1: Handwritten Digit Recognition (MNIST)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# Build model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

### Example 2: Text Classification

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential

# Sample data
texts = ["I love this product", "This is terrible", "Great service", ...]
labels = [1, 0, 1, ...]  # Binary classification

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Build model
model = Sequential([
    layers.Embedding(10000, 128, input_length=100),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(padded_sequences, labels, epochs=10, validation_split=0.2)
```

---

## Tools and Frameworks {#tools}

### TensorFlow/Keras

```bash
pip install tensorflow
```

**Pros**: 
- Production-ready
- Excellent documentation
- Strong ecosystem
- TensorBoard visualization

### PyTorch

```bash
pip install torch torchvision
```

**Pros**:
- Dynamic computation graphs
- Pythonic API
- Great for research
- Strong community

### JAX

```bash
pip install jax jaxlib
```

**Pros**:
- NumPy-like API
- Automatic differentiation
- GPU/TPU acceleration
- Functional programming style

---

## Best Practices

1. **Start Simple**: Begin with basic architectures
2. **Use Pre-trained Models**: Leverage transfer learning
3. **Monitor Training**: Use callbacks and TensorBoard
4. **Regularize**: Prevent overfitting with dropout, L2, etc.
5. **Normalize Data**: Always normalize inputs
6. **Batch Normalization**: Use in hidden layers
7. **Learning Rate**: Use learning rate scheduling
8. **Early Stopping**: Prevent overfitting
9. **Hyperparameter Tuning**: Systematically search space
10. **Version Control**: Track experiments and models

---

## Resources

- **Deep Learning Book**: deeplearningbook.org
- **Fast.ai**: fast.ai
- **TensorFlow Tutorials**: tensorflow.org/tutorials
- **PyTorch Tutorials**: pytorch.org/tutorials
- **Papers with Code**: paperswithcode.com

---

## Conclusion

Deep Learning enables solving complex problems that traditional ML cannot handle. Key takeaways:

1. **Understand Fundamentals**: Neural networks, backpropagation, optimization
2. **Choose Right Architecture**: Match problem to architecture type
3. **Regularize**: Prevent overfitting is crucial
4. **Use Transfer Learning**: Leverage pre-trained models
5. **Experiment**: Deep learning requires experimentation

Remember: Deep learning is powerful but requires data, computation, and careful tuning!

