"""
TensorFlow Basics Example
Demonstrates fundamental TensorFlow operations and concepts
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================================
# 1. Creating Tensors
# ============================================================================
print("\n" + "="*60)
print("1. Creating Tensors")
print("="*60)

# From Python lists
t1 = tf.constant([1, 2, 3, 4])
print(f"Constant tensor: {t1}")

# From NumPy arrays
arr = np.array([[1, 2], [3, 4]])
t2 = tf.constant(arr)
print(f"From NumPy:\n{t2.numpy()}")

# Zeros and ones
t3 = tf.zeros((3, 3))
t4 = tf.ones((2, 4))
print(f"Zeros:\n{t3.numpy()}")
print(f"Ones:\n{t4.numpy()}")

# Random tensors
t5 = tf.random.normal((2, 3), mean=0.0, stddev=1.0)
t6 = tf.random.uniform((2, 3), minval=0, maxval=10, dtype=tf.int32)
print(f"Normal distribution:\n{t5.numpy()}")
print(f"Uniform distribution:\n{t6.numpy()}")

# Range
t7 = tf.range(10)
print(f"Range: {t7.numpy()}")

# Variables (mutable)
var = tf.Variable([1.0, 2.0, 3.0])
var.assign([4.0, 5.0, 6.0])
print(f"Variable: {var.numpy()}")

# ============================================================================
# 2. Tensor Operations
# ============================================================================
print("\n" + "="*60)
print("2. Tensor Operations")
print("="*60)

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

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
print(f"Max: {max_all.numpy()}")
print(f"Min: {min_all.numpy()}")

# Reshaping
x = tf.constant([[1, 2, 3], [4, 5, 6]])
x_reshaped = tf.reshape(x, (3, 2))
print(f"Original shape: {x.shape}")
print(f"Reshaped:\n{x_reshaped.numpy()}")

# Broadcasting
a_broadcast = tf.constant([[1, 2, 3], [4, 5, 6]])
b_broadcast = tf.constant([10, 20, 30])
result = a_broadcast + b_broadcast
print(f"Broadcasting:\n{result.numpy()}")

# ============================================================================
# 3. Building a Simple Model
# ============================================================================
print("\n" + "="*60)
print("3. Building a Simple Model")
print("="*60)

from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data
X = tf.random.normal((1000, 20))
y = tf.random.uniform((1000,), minval=0, maxval=2, dtype=tf.int32)

# Build model using Sequential API
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
model.summary()

# ============================================================================
# 4. Training the Model
# ============================================================================
print("\n" + "="*60)
print("4. Training the Model")
print("="*60)

# Train model
history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X[:100], y[:100], verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ============================================================================
# 5. Using tf.data API
# ============================================================================
print("\n" + "="*60)
print("5. Using tf.data API")
print("="*60)

# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and batch
dataset = dataset.shuffle(buffer_size=1000).batch(32)

# Prefetch for performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Iterate through dataset
print("Iterating through dataset:")
for i, (batch_x, batch_y) in enumerate(dataset.take(3)):
    print(f"Batch {i+1}: x shape = {batch_x.shape}, y shape = {batch_y.shape}")

# ============================================================================
# 6. Custom Training Loop
# ============================================================================
print("\n" + "="*60)
print("6. Custom Training Loop")
print("="*60)

# Create a simple model
simple_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy()
train_acc_metric = keras.metrics.BinaryAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = simple_model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, simple_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, simple_model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

# Train for a few steps
print("Custom training loop:")
for step, (x_batch, y_batch) in enumerate(dataset.take(5)):
    loss_value = train_step(x_batch, y_batch)
    if step % 1 == 0:
        acc = train_acc_metric.result()
        print(f"Step {step}: Loss = {loss_value:.4f}, Accuracy = {acc:.4f}")
        train_acc_metric.reset_states()

# ============================================================================
# 7. Model Saving and Loading
# ============================================================================
print("\n" + "="*60)
print("7. Model Saving and Loading")
print("="*60)

# Save model
model.save('saved_model')
print("Model saved to 'saved_model' directory")

# Save weights only
model.save_weights('model_weights.h5')
print("Weights saved to 'model_weights.h5'")

# Load model
loaded_model = keras.models.load_model('saved_model')
print("Model loaded successfully")

# Load weights
model.load_weights('model_weights.h5')
print("Weights loaded successfully")

# ============================================================================
# 8. Using Callbacks
# ============================================================================
print("\n" + "="*60)
print("8. Using Callbacks")
print("="*60)

# Create callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'checkpoint_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )
]

print("Callbacks created:")
print("- EarlyStopping: Stop training if no improvement")
print("- ModelCheckpoint: Save best model")
print("- ReduceLROnPlateau: Reduce learning rate on plateau")

# ============================================================================
# 9. Graph Mode vs Eager Mode
# ============================================================================
print("\n" + "="*60)
print("9. Graph Mode vs Eager Mode")
print("="*60)

# Eager execution (default)
def eager_function(x, y):
    return tf.matmul(x, y)

result_eager = eager_function(a, b)
print(f"Eager execution result:\n{result_eager.numpy()}")

# Graph mode with @tf.function
@tf.function
def graph_function(x, y):
    return tf.matmul(x, y)

result_graph = graph_function(a, b)
print(f"Graph execution result:\n{result_graph.numpy()}")

print("\n" + "="*60)
print("TensorFlow Basics Example Completed!")
print("="*60)

