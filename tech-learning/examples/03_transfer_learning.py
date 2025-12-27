"""
Transfer Learning Example
Using pre-trained models for custom image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def create_transfer_learning_model(base_model_name='MobileNetV2', num_classes=5, input_shape=(224, 224, 3)):
    """
    Create a transfer learning model using pre-trained base
    
    Args:
        base_model_name: Name of pre-trained model ('VGG16', 'ResNet50', 'MobileNetV2')
        num_classes: Number of output classes
        input_shape: Input image shape
    """
    
    # Load pre-trained base model
    if base_model_name == 'VGG16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    
    # Preprocessing (for models that need it)
    if base_model_name == 'MobileNetV2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    else:
        x = inputs
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Custom classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

# Create model
model, base_model = create_transfer_learning_model(
    base_model_name='MobileNetV2',
    num_classes=5
)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

# Validation data (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Example: Assuming data is in 'data/train' directory with subdirectories for each class
# train_generator = train_datagen.flow_from_directory(
#     'data/train',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )
# 
# val_generator = val_datagen.flow_from_directory(
#     'data/train',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_transfer_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

# Train model (uncomment when data is available)
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     callbacks=callbacks,
#     verbose=1
# )

def fine_tune_model(model, base_model, num_unfreeze_layers=10):
    """
    Fine-tune the model by unfreezing some base model layers
    
    Args:
        model: The compiled model
        base_model: The base model used
        num_unfreeze_layers: Number of top layers to unfreeze
    """
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze all layers except the last num_unfreeze_layers
    for layer in base_model.layers[:-num_unfreeze_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Fine-tune model (uncomment when ready)
# model = fine_tune_model(model, base_model, num_unfreeze_layers=10)
# 
# history_finetune = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=val_generator,
#     callbacks=callbacks,
#     verbose=1
# )

print("\nTransfer learning model created successfully!")
print("To use this model:")
print("1. Prepare your dataset in 'data/train' directory")
print("2. Organize images into subdirectories by class")
print("3. Uncomment the training code")
print("4. Run the script")

