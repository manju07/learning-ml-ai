"""
U-Net for Image Segmentation
Semantic segmentation example using U-Net architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def create_unet(input_shape=(256, 256, 3), num_classes=2):
    """
    Create U-Net model for semantic segmentation
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of segmentation classes
    """
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
    
    # Block 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Expansive Path)
    # Block 6
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    # Block 7
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    # Block 8
    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    # Block 9
    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs, outputs)
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Intersection over Union (IoU) metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Create model
model = create_unet(input_shape=(256, 256, 3), num_classes=2)

# Compile model with combined loss
model.compile(
    optimizer='adam',
    loss=dice_loss,
    metrics=['accuracy', dice_coefficient, iou_metric]
)

# Display model architecture
model.summary()

# Example training (uncomment when data is available)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 
# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2
# )
# 
# # Assuming you have images and masks in separate directories
# # train_image_generator = train_datagen.flow_from_directory(
# #     'data/train/images',
# #     target_size=(256, 256),
# #     batch_size=16,
# #     class_mode=None
# # )
# # 
# # train_mask_generator = train_datagen.flow_from_directory(
# #     'data/train/masks',
# #     target_size=(256, 256),
# #     batch_size=16,
# #     class_mode=None
# # )
# 
# # Combine generators
# def combined_generator(image_gen, mask_gen):
#     while True:
#         images = image_gen.next()
#         masks = mask_gen.next()
#         yield images, masks
# 
# # train_gen = combined_generator(train_image_generator, train_mask_generator)
# 
# # Train model
# history = model.fit(
#     train_gen,
#     steps_per_epoch=100,
#     epochs=50,
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=10),
#         keras.callbacks.ModelCheckpoint('best_unet_model.h5', save_best_only=True)
#     ]
# )

def visualize_predictions(model, images, masks, num_samples=5):
    """Visualize segmentation predictions"""
    predictions = model.predict(images[:num_samples])
    predicted_masks = np.argmax(predictions, axis=-1)
    
    plt.figure(figsize=(15, num_samples * 3))
    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(predicted_masks[i], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_predictions.png')
    print("Segmentation visualization saved as 'segmentation_predictions.png'")

print("\nU-Net model created successfully!")
print("To use this model:")
print("1. Prepare your dataset with images and corresponding masks")
print("2. Organize data in 'data/train/images' and 'data/train/masks'")
print("3. Uncomment the training code")
print("4. Run the script")

