# Convolutional Neural Networks (CNN): Complete Guide

## Table of Contents
1. [Introduction to CNNs](#introduction-to-cnns)
2. [CNN Fundamentals](#cnn-fundamentals)
3. [Convolutional Layers](#convolutional-layers)
4. [Pooling Layers](#pooling-layers)
5. [CNN Architectures](#cnn-architectures)
6. [Transfer Learning with CNNs](#transfer-learning-with-cnns)
7. [Object Detection](#object-detection)
8. [Image Segmentation](#image-segmentation)
9. [Practical Examples](#practical-examples)
10. [Advanced Topics](#advanced-topics)

---

## Introduction to CNNs

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data such as images. They use convolutional operations to automatically learn spatial hierarchies of features.

### Why CNNs for Images?

- **Translation Invariance**: Detect features regardless of position
- **Parameter Sharing**: Same filters used across image (reduces parameters)
- **Spatial Hierarchy**: Learn edges → shapes → objects
- **Efficiency**: Much fewer parameters than fully connected networks

### Key Applications
- Image Classification
- Object Detection
- Image Segmentation
- Face Recognition
- Medical Image Analysis
- Autonomous Vehicles

---

## CNN Fundamentals

### Basic CNN Structure

```
Input Image → Convolution → Activation → Pooling → Convolution → Activation → Pooling → Fully Connected → Output
```

### Convolution Operation Explained

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Simple convolution example
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Edge detection filter
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply convolution
convolved = signal.convolve2d(image, kernel, mode='valid')
print("Original Image:")
print(image)
print("\nKernel (Edge Detector):")
print(kernel)
print("\nConvolved Output:")
print(convolved)
```

### Visualizing Convolution

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Create a simple image (5x5)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]).reshape(1, 5, 5, 1).astype('float32')

# Create a vertical edge detector kernel
kernel = np.array([
    [[[-1]], [[0]], [[1]]],
    [[[-1]], [[0]], [[1]]],
    [[[-1]], [[0]], [[1]]]
]).astype('float32')

# Apply convolution
conv_layer = layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding='valid',
    use_bias=False,
    input_shape=(5, 5, 1)
)
conv_layer.build(input_shape=(None, 5, 5, 1))
conv_layer.set_weights([kernel])

output = conv_layer(image)
print("Output shape:", output.shape)
print("Output values:")
print(output.numpy().squeeze())
```

---

## Convolutional Layers

### Basic Convolutional Layer

```python
from tensorflow.keras import layers, Sequential

# Single convolutional layer
conv_layer = layers.Conv2D(
    filters=32,           # Number of filters
    kernel_size=(3, 3),   # Filter size
    strides=(1, 1),       # Step size
    padding='same',       # 'same' or 'valid'
    activation='relu',
    input_shape=(28, 28, 1)
)

# Output size calculation:
# For padding='same': output_size = input_size / stride
# For padding='valid': output_size = (input_size - kernel_size + 1) / stride
```

### Multiple Convolutional Layers

```python
model = Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

### Depthwise Separable Convolution

```python
# More efficient than standard convolution
model = Sequential([
    layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.SeparableConv2D(64, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])
```

### Dilated Convolution

```python
# Increases receptive field without increasing parameters
dilated_conv = layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    dilation_rate=(2, 2),  # Adds gaps in kernel
    activation='relu'
)
```

---

## Pooling Layers

### Max Pooling

```python
# Reduces spatial dimensions, keeps most important features
max_pool = layers.MaxPooling2D(
    pool_size=(2, 2),  # 2x2 window
    strides=(2, 2),    # Step size
    padding='valid'
)

# Example: 4x4 → 2x2
input_tensor = np.array([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
]).reshape(1, 4, 4, 1)

pool_layer = layers.MaxPooling2D(pool_size=(2, 2))
output = pool_layer(input_tensor)
print("Max Pooling Output:")
print(output.numpy().squeeze())
# Output: [[6, 8], [14, 16]]
```

### Average Pooling

```python
# Takes average instead of max
avg_pool = layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
)
```

### Global Pooling

```python
# Reduces entire feature map to single value per channel
global_max_pool = layers.GlobalMaxPooling2D()
global_avg_pool = layers.GlobalAveragePooling2D()

# Useful before final classification layer
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.GlobalAveragePooling2D(),  # (224, 224, 64) → (64,)
    layers.Dense(10, activation='softmax')
])
```

---

## CNN Architectures

### 1. LeNet-5 (Classic Architecture)

```python
def create_lenet5(input_shape=(32, 32, 1), num_classes=10):
    model = Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=input_shape),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

lenet = create_lenet5()
lenet.summary()
```

### 2. AlexNet

```python
def create_alexnet(input_shape=(227, 227, 3), num_classes=1000):
    model = Sequential([
        # First conv block
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.BatchNormalization(),
        
        # Second conv block
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.BatchNormalization(),
        
        # Third conv block
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 3. VGG (Very Deep Convolutional Networks)

```python
def create_vgg16(input_shape=(224, 224, 3), num_classes=1000):
    model = Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 4. ResNet (Residual Networks)

```python
from tensorflow.keras import Model

def residual_block(x, filters, kernel_size=3, stride=1):
    """Residual block with skip connection"""
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_resnet18(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, x)
    return model
```

### 5. Inception Network

```python
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5, filters_pool):
    """Inception module with multiple parallel paths"""
    
    # 1x1 conv path
    path1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    # 3x3 conv path
    path2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(path2)
    
    # 5x5 conv path
    path3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(path3)
    
    # Pool path
    path4 = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)
    path4 = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate all paths
    output = layers.Concatenate()([path1, path2, path3, path4])
    return output
```

---

## Transfer Learning with CNNs

### Using Pre-trained Models

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras import layers, Model

# Method 1: Feature Extraction (Frozen Base)
def create_feature_extractor(base_model_name='VGG16', num_classes=10):
    # Load pre-trained model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classifier
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

model = create_feature_extractor('ResNet50', num_classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Fine-tuning Pre-trained Models

```python
def fine_tune_model(base_model_name='VGG16', num_classes=10):
    # Load base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze early layers, unfreeze later layers
    base_model.trainable = True
    for layer in base_model.layers[:-4]:  # Freeze all except last 4 layers
        layer.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

## Object Detection

### YOLO (You Only Look Once) - Simplified

```python
def create_yolo_like_model(input_shape=(416, 416, 3), num_classes=20, num_boxes=5):
    """Simplified YOLO-like architecture"""
    
    # Backbone (Darknet-like)
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # More layers...
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Detection head
    # Output: (grid_h, grid_w, num_boxes * (5 + num_classes))
    # 5 = [x, y, w, h, confidence]
    x = layers.Conv2D(num_boxes * (5 + num_classes), (1, 1))(x)
    
    model = Model(inputs, x)
    return model
```

### Using Pre-trained Object Detection Models

```python
# Using TensorFlow Object Detection API
import tensorflow_hub as hub

# Load pre-trained model
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Run detection
detections = detector(image)
```

---

## Image Segmentation

### U-Net Architecture

```python
def create_unet(input_shape=(256, 256, 3), num_classes=2):
    """U-Net for semantic segmentation"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder (Expansive Path)
    # Block 5
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    # Block 6
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    # Block 7
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    
    model = Model(inputs, outputs)
    return model
```

### FCN (Fully Convolutional Network)

```python
def create_fcn(input_shape=(224, 224, 3), num_classes=21):
    """Fully Convolutional Network"""
    
    # Use VGG16 as backbone
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Get feature maps at different scales
    block4_pool = base_model.get_layer('block4_pool').output
    block5_pool = base_model.get_layer('block5_pool').output
    
    # FCN-32s: Direct upsampling
    x = layers.Conv2D(num_classes, (1, 1))(block5_pool)
    x = layers.UpSampling2D((32, 32), interpolation='bilinear')(x)
    
    model = Model(base_model.input, x)
    return model
```

---

## Practical Examples

### Example 1: CIFAR-10 Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Build CNN
model = keras.Sequential([
    # First block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)
```

### Example 2: Transfer Learning for Custom Dataset

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base
base_model.trainable = False

# Add classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # 5 classes
])

model.compile(
    optimizer='adam',
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

## Advanced Topics

### Attention Mechanisms

```python
def attention_block(x):
    """Self-attention mechanism"""
    # Query, Key, Value
    query = layers.Conv2D(64, (1, 1))(x)
    key = layers.Conv2D(64, (1, 1))(x)
    value = layers.Conv2D(64, (1, 1))(x)
    
    # Attention scores
    attention = layers.Multiply()([query, key])
    attention = layers.Activation('softmax')(attention)
    
    # Apply attention
    output = layers.Multiply()([attention, value])
    return output
```

### Custom Loss Functions

```python
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced datasets"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed
```

### Model Ensembling

```python
def ensemble_predict(models, x):
    """Average predictions from multiple models"""
    predictions = []
    for model in models:
        pred = model.predict(x)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

---

## Best Practices

1. **Data Augmentation**: Always use augmentation for small datasets
2. **Batch Normalization**: Use after conv layers
3. **Transfer Learning**: Start with pre-trained models
4. **Learning Rate**: Use learning rate scheduling
5. **Regularization**: Dropout, L2, data augmentation
6. **Architecture**: Start simple, add complexity gradually
7. **Monitoring**: Use TensorBoard to monitor training
8. **Early Stopping**: Prevent overfitting
9. **Ensemble**: Combine multiple models for better performance
10. **Test Time Augmentation**: Apply augmentation during inference

---

## Resources

- **Papers**: 
  - LeNet-5 (1998)
  - AlexNet (2012)
  - VGG (2014)
  - ResNet (2015)
  - YOLO (2016)
  - U-Net (2015)

- **Datasets**:
  - ImageNet
  - CIFAR-10/100
  - COCO
  - Pascal VOC

- **Tools**:
  - TensorFlow/Keras
  - PyTorch
  - TensorFlow Object Detection API
  - Detectron2

---

## Conclusion

CNNs are powerful tools for computer vision tasks. Key takeaways:

1. **Understand Convolution**: Core operation of CNNs
2. **Use Pre-trained Models**: Transfer learning saves time and resources
3. **Data Augmentation**: Critical for small datasets
4. **Architecture Matters**: Choose architecture based on problem
5. **Regularize**: Prevent overfitting with various techniques

Remember: CNNs excel at learning spatial hierarchies in images. Start with transfer learning and fine-tune for your specific task!

