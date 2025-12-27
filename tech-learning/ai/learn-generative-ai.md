# Generative AI: Complete Guide

## Table of Contents
1. [Introduction to Generative AI](#introduction)
2. [Generative Adversarial Networks (GANs)](#gans)
3. [Variational Autoencoders (VAEs)](#vaes)
4. [Diffusion Models](#diffusion)
5. [Stable Diffusion](#stable-diffusion)
6. [Transformer-based Generators](#transformer-generators)
7. [Text-to-Image Models](#text-to-image)
8. [Image-to-Image Translation](#image-to-image)
9. [Practical Examples](#examples)
10. [Best Practices](#best-practices)

---

## Introduction to Generative AI {#introduction}

Generative AI creates new content (images, text, audio, video) that resembles training data. Unlike discriminative models that classify, generative models learn the data distribution.

### Types of Generative Models

- **GANs**: Adversarial training with generator and discriminator
- **VAEs**: Probabilistic encoder-decoder architecture
- **Diffusion Models**: Iterative denoising process
- **Autoregressive Models**: Generate sequentially (GPT, PixelRNN)
- **Flow-based Models**: Learn invertible transformations

### Applications

- **Image Generation**: Create realistic images
- **Text Generation**: Write stories, code, articles
- **Image Editing**: Style transfer, inpainting, super-resolution
- **Data Augmentation**: Generate synthetic training data
- **Anomaly Detection**: Identify outliers
- **Drug Discovery**: Generate molecular structures

---

## Generative Adversarial Networks (GANs) {#gans}

### GAN Architecture

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Generator(keras.Model):
    """Generator network"""
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.model = keras.Sequential([
            layers.Dense(256, input_dim=latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(28 * 28, activation='tanh'),
            layers.Reshape((28, 28, 1))
        ])
    
    def call(self, z):
        return self.model(z)

class Discriminator(keras.Model):
    """Discriminator network"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, img):
        return self.model(img)
```

### GAN Training

```python
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim=100):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors, training=True)
        
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # Label smoothing
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors, training=True), training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Usage
generator = Generator()
discriminator = Discriminator()
gan = GAN(generator, discriminator)

gan.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy()
)

gan.fit(train_dataset, epochs=50)
```

### DCGAN (Deep Convolutional GAN)

```python
class DCGAN_Generator(keras.Model):
    """DCGAN Generator with transposed convolutions"""
    def __init__(self, latent_dim=100):
        super(DCGAN_Generator, self).__init__()
        
        self.model = keras.Sequential([
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            
            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])
    
    def call(self, z):
        return self.model(z)
```

---

## Variational Autoencoders (VAEs) {#vaes}

### VAE Architecture

```python
class VAE(keras.Model):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(16, activation='relu')
        ])
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')
        ])
    
    def encode(self, x):
        encoded = self.encoder(x)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

# Loss function
def vae_loss(x, reconstructed, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(x, reconstructed),
            axis=(1, 2)
        )
    )
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )
    )
    return reconstruction_loss + kl_loss
```

---

## Diffusion Models {#diffusion}

### Denoising Diffusion Probabilistic Model (DDPM)

```python
class DiffusionModel(keras.Model):
    def __init__(self, image_size=32, widths=[32, 64, 96, 128], block_depth=2):
        super(DiffusionModel, self).__init__()
        
        self.normalizer = layers.Normalization()
        self.network = self.build_network(image_size, widths, block_depth)
        self.image_size = image_size
    
    def build_network(self, image_size, widths, block_depth):
        input_shape = (image_size, image_size, 3)
        inputs = keras.Input(shape=input_shape)
        x = self.normalizer(inputs)
        
        # U-Net style architecture
        x = layers.Conv2D(widths[0], kernel_size=3, padding='same')(x)
        
        # Downsampling
        skips = []
        for width in widths[:-1]:
            x = self.down_block(x, width, block_depth)
            skips.append(x)
        
        # Middle
        x = self.mid_block(x, widths[-1], block_depth)
        
        # Upsampling
        for width in reversed(widths[:-1]):
            x = self.up_block(x, skips.pop(), width, block_depth)
        
        # Output
        x = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)
        return keras.Model(inputs, x, name="unet")
    
    def down_block(self, x, width, block_depth):
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        for _ in range(block_depth):
            x = self.residual_block(x, width)
        return layers.MaxPooling2D(pool_size=2)(x)
    
    def up_block(self, x, skip, width, block_depth):
        x = layers.UpSampling2D(size=2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        for _ in range(block_depth):
            x = self.residual_block(x, width)
        return x
    
    def residual_block(self, x, width):
        input_tensor = x
        x = layers.Conv2D(width, kernel_size=3, padding='same', activation='swish')(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        return layers.Add()([input_tensor, x])
    
    def call(self, x, training=False):
        return self.network(x, training=training)
```

### Diffusion Training

```python
class DiffusionTrainer:
    def __init__(self, model, image_size=32, max_signal_rate=0.95, min_signal_rate=0.02):
        self.model = model
        self.image_size = image_size
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
    
    def get_diffusion_schedule(self, diffusion_times):
        """Get noise schedule"""
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)
        
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        return signal_rates, noise_rates
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Denoise images"""
        pred_noises = self.model([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images
    
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        
        # Sample random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1),
            minval=0.0, maxval=1.0
        )
        
        # Get noise schedule
        signal_rates, noise_rates = self.get_diffusion_schedule(diffusion_times)
        
        # Sample noise
        noises = tf.random.normal(shape=tf.shape(images))
        
        # Create noisy images
        noisy_images = signal_rates * images + noise_rates * noises
        
        # Predict noise
        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)
            loss = noise_loss + image_loss
        
        # Update weights
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        return {"loss": loss, "noise_loss": noise_loss, "image_loss": image_loss}
```

---

## Stable Diffusion {#stable-diffusion}

### Using Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image from text
prompt = "A beautiful sunset over mountains, highly detailed, 4k"
image = pipe(prompt).images[0]
image.save("generated_image.png")

# With negative prompt
negative_prompt = "blurry, low quality"
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# Image-to-image
from diffusers import StableDiffusionImg2ImgPipeline

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# Load input image
from PIL import Image
init_image = Image.open("input.jpg")

# Transform image
image = img2img_pipe(
    prompt="A fantasy landscape",
    image=init_image,
    strength=0.75  # How much to transform (0-1)
).images[0]
```

### Fine-tuning Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
import torch

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# Fine-tune on custom dataset
# (Training loop here)
```

---

## Transformer-based Generators {#transformer-generators}

### GPT-style Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt = "The future of AI is"
inputs = tokenizer.encode(prompt, return_tensors='pt')

outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Image Generation with DALL-E Style

```python
# Using OpenAI DALL-E API (example)
import openai

response = openai.Image.create(
    prompt="A futuristic city at sunset",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']
```

---

## Text-to-Image Models {#text-to-image}

### Using Hugging Face Diffusers

```python
from diffusers import DiffusionPipeline
import torch

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate
prompt = "A cyberpunk cityscape at night, neon lights, rain"
image = pipe(prompt).images[0]
image.save("cyberpunk_city.png")

# With control
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# Use canny edge detection for control
from PIL import Image
import numpy as np
import cv2

image = Image.open("input.jpg")
image = np.array(image)
canny_image = cv2.Canny(image, 100, 200)
canny_image = Image.fromarray(canny_image)

output = pipe(
    prompt,
    image=canny_image,
    num_inference_steps=20
).images[0]
```

---

## Image-to-Image Translation {#image-to-image}

### CycleGAN

```python
class CycleGAN:
    """CycleGAN for unpaired image-to-image translation"""
    def __init__(self):
        self.generator_A_to_B = self.build_generator()
        self.generator_B_to_A = self.build_generator()
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()
    
    def build_generator(self):
        """U-Net style generator"""
        # Implementation here
        pass
    
    def build_discriminator(self):
        """PatchGAN discriminator"""
        # Implementation here
        pass
    
    def cycle_loss(self, real, reconstructed):
        """Cycle consistency loss"""
        return tf.reduce_mean(tf.abs(real - reconstructed))
    
    def identity_loss(self, real, same):
        """Identity loss"""
        return tf.reduce_mean(tf.abs(real - same))
```

### Style Transfer

```python
import tensorflow_hub as hub

# Load pre-trained style transfer model
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
```

---

## Practical Examples {#examples}

### Example 1: Generate Faces with GAN

```python
# Train DCGAN on CelebA dataset
# Generate new faces
latent_vectors = tf.random.normal(shape=(16, 100))
generated_faces = generator(latent_vectors)

# Display
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_faces[i].numpy().squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
```

### Example 2: Text-to-Image Generation

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompts = [
    "A serene lake surrounded by mountains",
    "A futuristic robot in a cyberpunk city",
    "A cozy coffee shop interior"
]

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=50).images[0]
    image.save(f"{prompt.replace(' ', '_')}.png")
```

---

## Best Practices {#best-practices}

1. **Start with Pre-trained Models**: Use Stable Diffusion, GPT, etc.
2. **Monitor Training**: Track losses, use TensorBoard
3. **Use Appropriate Losses**: GAN loss, VAE loss, diffusion loss
4. **Regularize**: Use batch normalization, dropout
5. **Experiment**: Try different architectures and hyperparameters
6. **Evaluate Quality**: Use FID, IS metrics for images
7. **Handle Mode Collapse**: In GANs, use techniques like spectral normalization

---

## Resources

- **Papers**: 
  - GAN (2014)
  - VAE (2013)
  - Diffusion Models (2020)
  - Stable Diffusion (2022)
- **Libraries**: 
  - diffusers (Hugging Face)
  - TensorFlow/PyTorch
  - GAN libraries

---

## Conclusion

Generative AI enables creating new content. Key takeaways:

1. **Choose Right Model**: GANs for images, VAEs for latent space, Diffusion for quality
2. **Use Pre-trained**: Start with Stable Diffusion, GPT, etc.
3. **Fine-tune**: Adapt models to your domain
4. **Evaluate**: Use appropriate metrics
5. **Experiment**: Generative AI requires experimentation

Remember: Generative models are powerful but require careful training and evaluation!

