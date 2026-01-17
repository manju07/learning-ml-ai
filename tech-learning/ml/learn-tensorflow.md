# TensorFlow: Complete Guide with Examples

## Table of Contents
1. [Introduction to TensorFlow](#introduction-to-tensorflow)
2. [TensorFlow Fundamentals](#tensorflow-fundamentals)
3. [Tensors and Operations](#tensors-and-operations)
4. [Building Models](#building-models)
5. [Training and Evaluation](#training-and-evaluation)
6. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
7. [Custom Layers and Models](#custom-layers-and-models)
8. [Callbacks and Monitoring](#callbacks-and-monitoring)
9. [Distributed Training](#distributed-training)
10. [TensorBoard Visualization](#tensorboard-visualization)
11. [Model Saving and Loading](#model-saving-and-loading)
12. [TensorFlow Serving](#tensorflow-serving)
13. [Advanced Topics](#advanced-topics)
14. [Practical Examples](#practical-examples)

---

## Introduction to TensorFlow

TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem for building and deploying ML models.

### Key Features
- **Eager Execution**: Immediate evaluation of operations
- **Keras Integration**: High-level API for rapid prototyping
- **Graph Execution**: Optimized computation graphs
- **GPU/TPU Support**: Accelerated computing
- **Production Ready**: TensorFlow Serving, TensorFlow Lite
- **Cross-platform**: Windows, Linux, macOS, Mobile, Web

### TensorFlow 2.x vs 1.x

**TensorFlow 2.x Improvements**:
- Eager execution by default
- Keras as primary high-level API
- Simplified API
- Better performance
- Improved error messages

### Installation

```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow-gpu

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## TensorFlow Fundamentals

### Basic Concepts

```python
import tensorflow as tf
import numpy as np

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
```

### Eager Execution

```python
# Eager execution (default in TF 2.x)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Operations execute immediately
c = tf.matmul(a, b)
print("Result:")
print(c.numpy())
```

### Graph Mode (for optimization)

```python
@tf.function
def compute(x, y):
    return tf.matmul(x, y)

# First call traces the function
result = compute(a, b)
print(result.numpy())
```

---

## Tensors and Operations

### Creating Tensors

```python
# Different ways to create tensors
# 1. From Python lists
t1 = tf.constant([1, 2, 3, 4])
print(f"t1: {t1}")

# 2. From NumPy arrays
arr = np.array([1, 2, 3, 4])
t2 = tf.constant(arr)
print(f"t2: {t2}")

# 3. Zeros and ones
t3 = tf.zeros((3, 3))
t4 = tf.ones((2, 4))
print(f"Zeros:\n{t3.numpy()}")
print(f"Ones:\n{t4.numpy()}")

# 4. Random tensors
t5 = tf.random.normal((2, 3), mean=0.0, stddev=1.0)
t6 = tf.random.uniform((2, 3), minval=0, maxval=10, dtype=tf.int32)
print(f"Normal:\n{t5.numpy()}")
print(f"Uniform:\n{t6.numpy()}")

# 5. Range
t7 = tf.range(10)
print(f"Range: {t7.numpy()}")

# 6. Variables (mutable tensors)
var = tf.Variable([1.0, 2.0, 3.0])
var.assign([4.0, 5.0, 6.0])
print(f"Variable: {var.numpy()}")
```

### Tensor Operations

```python
# Basic operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise operations
add = tf.add(a, b)
subtract = tf.subtract(a, b)
multiply = tf.multiply(a, b)
divide = tf.divide(a, b)

print(f"Addition:\n{add.numpy()}")
print(f"Subtraction:\n{subtract.numpy()}")
print(f"Multiplication:\n{multiply.numpy()}")
print(f"Division:\n{divide.numpy()}")

# Matrix operations
matmul = tf.matmul(a, b)
print(f"Matrix multiplication:\n{matmul.numpy()}")

# Reduction operations
sum_all = tf.reduce_sum(a)
mean_all = tf.reduce_mean(a)
max_all = tf.reduce_max(a)
min_all = tf.reduce_min(a)

print(f"Sum: {sum_all.numpy()}")
print(f"Mean: {mean_all.numpy()}")
print(f"Max: {max_max.numpy()}")
print(f"Min: {min_all.numpy()}")

# Reshaping
x = tf.constant([[1, 2, 3], [4, 5, 6]])
x_reshaped = tf.reshape(x, (3, 2))
print(f"Original shape: {x.shape}")
print(f"Reshaped:\n{x_reshaped.numpy()}")

# Transpose
x_transposed = tf.transpose(x)
print(f"Transposed:\n{x_transposed.numpy()}")
```

### Broadcasting

```python
# Broadcasting example
a = tf.constant([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
b = tf.constant([10, 20, 30])            # Shape: (3,)

# Broadcasting adds b to each row of a
result = a + b
print(f"Broadcasting result:\n{result.numpy()}")
```

---

## Building Models

### Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

### Functional API

```python
# More flexible than Sequential API
inputs = keras.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

### Model Subclassing

```python
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(32, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)

model = MyModel()
model.build(input_shape=(None, 784))
model.summary()
```

### Common Layers

```python
# Dense (Fully Connected)
dense = layers.Dense(128, activation='relu', use_bias=True)

# Convolutional
conv2d = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
conv1d = layers.Conv1D(64, 3, activation='relu')

# Pooling
max_pool = layers.MaxPooling2D((2, 2))
avg_pool = layers.AveragePooling2D((2, 2))
global_pool = layers.GlobalAveragePooling2D()

# Recurrent
lstm = layers.LSTM(64, return_sequences=True)
gru = layers.GRU(64)
rnn = layers.SimpleRNN(64)

# Normalization
batch_norm = layers.BatchNormalization()
layer_norm = layers.LayerNormalization()

# Regularization
dropout = layers.Dropout(0.5)
spatial_dropout = layers.SpatialDropout2D(0.5)

# Reshaping
flatten = layers.Flatten()
reshape = layers.Reshape((28, 28))
```

---

## Training and Evaluation

### Compiling Models

```python
# Compile with optimizer, loss, and metrics
model.compile(
    optimizer='adam',  # or keras.optimizers.Adam(learning_rate=0.001)
    loss='sparse_categorical_crossentropy',  # or keras.losses.SparseCategoricalCrossentropy()
    metrics=['accuracy']  # or [keras.metrics.Accuracy()]
)

# Custom optimizer
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# Custom loss function
def custom_loss(y_true, y_pred):
    return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
```

### Training Models

```python
# Basic training
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# With validation data
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)

# With data generator
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 32,
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(x_val) // 32,
    verbose=1
)
```

### Evaluation

```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Predict on single sample
single_prediction = model.predict(x_test[0:1])
print(f"Prediction: {single_prediction}")
```

### Custom Training Loop

```python
# For more control over training
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value = train_step(x_batch, y_batch)
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss_value:.4f}")
    
    # Validation
    for x_batch, y_batch in val_dataset:
        test_step(x_batch, y_batch)
    
    train_acc = train_acc_metric.result()
    val_acc = val_acc_metric.result()
    print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # Reset metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
```

---

## Data Loading and Preprocessing

### Loading Built-in Datasets

```python
# MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# CIFAR-100
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# IMDB (for text)
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

### tf.data API

```python
# Create dataset from NumPy arrays
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Shuffle and batch
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)

# Prefetch for performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Map transformations
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

dataset = dataset.map(preprocess)

# Iterate
for batch_x, batch_y in dataset:
    # Process batch
    pass
```

### Image Data Generators

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Load from directory
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load from DataFrame
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='data/images',
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Text Preprocessing

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Convert to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Word embeddings
embedding_layer = layers.Embedding(
    input_dim=10000,
    output_dim=128,
    input_length=100
)
```

---

## Custom Layers and Models

### Custom Layer

```python
class MyDenseLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(MyDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(MyDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(MyDenseLayer, self).get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config

# Use custom layer
model = keras.Sequential([
    MyDenseLayer(64, activation='relu', input_shape=(784,)),
    MyDenseLayer(32, activation='relu'),
    MyDenseLayer(10, activation='softmax')
])
```

### Custom Model

```python
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
    
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, (3, 3), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        if input_shape[-1] != self.filters:
            self.shortcut = layers.Conv2D(self.filters, (1, 1))
        else:
            self.shortcut = lambda x: x
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        shortcut = self.shortcut(inputs)
        x = tf.add(x, shortcut)
        return tf.nn.relu(x)

# Use in model
model = keras.Sequential([
    layers.Conv2D(64, (7, 7), input_shape=(224, 224, 3)),
    ResidualBlock(64),
    ResidualBlock(128),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])
```

### Custom Loss Function

```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Use custom loss
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)
```

### Custom Metric

```python
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int64)
        
        tp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 1), tf.float32))
        fp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 0), tf.float32))
        fn = tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 1), tf.float32))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())
        return f1
    
    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Use custom metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', F1Score()]
)
```

---

## Callbacks and Monitoring

### Built-in Callbacks

```python
# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# CSV logger
csv_logger = keras.callbacks.CSVLogger('training.log')

# TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Use callbacks
model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint, reduce_lr, tensorboard]
)
```

### Custom Callback

```python
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Training started!")
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} starting...")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch}: loss = {logs['loss']:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} ended: loss = {logs['loss']:.4f}, acc = {logs['accuracy']:.4f}")
    
    def on_train_end(self, logs=None):
        print("Training completed!")

# Use custom callback
model.fit(x_train, y_train, epochs=10, callbacks=[CustomCallback()])
```

---

## Distributed Training

### Multi-GPU Training

```python
# Strategy for multi-GPU
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

# Build and compile model within strategy scope
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training automatically uses all GPUs
model.fit(x_train, y_train, epochs=10, batch_size=64)
```

### TPU Training

```python
# Connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Build model within TPU strategy
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=128)
```

---

## TensorBoard Visualization

### Basic Usage

```python
# Start TensorBoard (in terminal)
# tensorboard --logdir=./logs

# Create callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Train with TensorBoard
model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback]
)
```

### Custom Logging

```python
import datetime

# Create log directory with timestamp
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Log custom metrics
class CustomTensorBoard(keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        # Log custom scalar
        tf.summary.scalar('custom_metric', logs.get('custom_metric', 0), step=epoch)
        super().on_epoch_end(epoch, logs)
```

---

## Model Saving and Loading

### Save Entire Model

```python
# Save model (HDF5 format)
model.save('my_model.h5')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save model (SavedModel format - recommended)
model.save('my_model')

# Load SavedModel
loaded_model = keras.models.load_model('my_model')
```

### Save Weights Only

```python
# Save weights
model.save_weights('my_weights.h5')

# Load weights (model architecture must be same)
model.load_weights('my_weights.h5')
```

### Save Architecture Only

```python
# Save architecture as JSON
json_config = model.to_json()
with open('model_config.json', 'w') as f:
    f.write(json_config)

# Load architecture
with open('model_config.json', 'r') as f:
    json_config = f.read()
model = keras.models.model_from_json(json_config)

# Save architecture as YAML
yaml_config = model.to_yaml()
with open('model_config.yaml', 'w') as f:
    f.write(yaml_config)

# Load from YAML
with open('model_config.yaml', 'r') as f:
    yaml_config = f.read()
model = keras.models.model_from_yaml(yaml_config)
```

### Checkpointing

```python
# Save checkpoints during training
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint_callback])

# Load from checkpoint
model.load_weights('training/cp-0010.ckpt')
```

---

## TensorFlow Serving

### Save Model for Serving

```python
# Save model in SavedModel format
model.save('serving_model/1', save_format='tf')

# Or export explicitly
tf.saved_model.save(model, 'serving_model/1')
```

### Serve with Docker

```bash
# Pull TensorFlow Serving image
docker pull tensorflow/serving

# Run container
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/serving_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

### Make Predictions

```python
import requests
import json

# Prepare data
data = {
    "instances": x_test[:3].tolist()
}

# Make request
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=json.dumps(data)
)

predictions = json.loads(response.text)['predictions']
print(predictions)
```

---

## Advanced Topics

### Mixed Precision Training

```python
# Enable mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Build model (output layer should use float32)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32')  # Keep output in float32
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
```

### Gradient Clipping

```python
# Clip gradients during training
optimizer = keras.optimizers.Adam(clipnorm=1.0)
# or
optimizer = keras.optimizers.Adam(clipvalue=0.5)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Learning Rate Scheduling

```python
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# Or use built-in schedules
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Model Pruning

```python
import tensorflow_model_optimization as tfmot

# Prune model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.90,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile and train
model_for_pruning.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_for_pruning.fit(x_train, y_train, epochs=10)
```

---

## Practical Examples

### Example 1: Complete Training Pipeline

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
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

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Example 2: Transfer Learning Pipeline

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add classifier
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train
model.fit(train_generator, epochs=10)
```

---

## Best Practices

1. **Use tf.data**: For efficient data loading and preprocessing
2. **Enable GPU**: Use GPU for faster training
3. **Use Callbacks**: Monitor training and save checkpoints
4. **Validate**: Always use validation data
5. **Regularize**: Use dropout, L2, batch normalization
6. **Save Models**: Save both architecture and weights
7. **Use TensorBoard**: Visualize training progress
8. **Optimize**: Use @tf.function for graph mode
9. **Mixed Precision**: Use for faster training on modern GPUs
10. **Version Control**: Track experiments and hyperparameters

---

## Resources

- **Official Documentation**: tensorflow.org
- **TensorFlow Tutorials**: tensorflow.org/tutorials
- **Keras Guide**: keras.io
- **TensorFlow Hub**: tfhub.dev
- **TensorFlow Model Garden**: github.com/tensorflow/models

---

## Conclusion

TensorFlow provides a comprehensive ecosystem for building and deploying ML models. Key takeaways:

1. **Start with Keras**: Use high-level API for rapid prototyping
2. **Use tf.data**: Efficient data pipelines
3. **Monitor Training**: Use callbacks and TensorBoard
4. **Save Models**: Use SavedModel format for production
5. **Optimize**: Use distributed training and mixed precision

Remember: TensorFlow is powerful and flexible. Start simple, then explore advanced features as needed!

