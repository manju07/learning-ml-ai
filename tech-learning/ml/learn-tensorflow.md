# TensorFlow & Keras: Comprehensive Deep Learning Guide

## Table of Contents
1. [Introduction: Eager vs Graph Execution](#1-introduction-eager-vs-graph-execution)
2. [Tensors: tf.Tensor, tf.Variable, tf.constant](#2-tensors)
3. [tf.function: Graph Compilation, Tracing, Retracing](#3-tffunction)
4. [Keras API: Sequential, Functional, Subclassing](#4-keras-api)
5. [Common Layers: Dense, Conv2D, LSTM, Attention and More](#5-common-layers)
6. [Custom Layers, Models, Loss Functions, Metrics](#6-custom-components)
7. [Training: compile, fit, evaluate](#7-training)
8. [Callbacks: Built-in and Custom](#8-callbacks)
9. [tf.data: Efficient Data Pipelines](#9-tfdata)
10. [Distributed Training: MirroredStrategy, TPUStrategy](#10-distributed-training)
11. [Mixed Precision](#11-mixed-precision)
12. [SavedModel, TF-Lite, TF-Serving](#12-saving-and-deployment)
13. [TF Hub for Transfer Learning](#13-tf-hub)
14. [TF Probability Basics](#14-tf-probability)
15. [Keras Tuner](#15-keras-tuner)
16. [Full Examples: Image Classification, Text Classification, Custom Loop](#16-full-examples)
17. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
18. [Production Deployment Notes](#production-deployment-notes)

---

## 1. Introduction: Eager vs Graph Execution

TensorFlow 2.x uses **eager execution** by default: operations run immediately and return concrete values, just like NumPy. This makes debugging intuitive.

**Graph execution** (via `@tf.function`) compiles Python code into a portable, optimized computation graph. It enables optimizations (op fusion, constant folding), enables distribution across devices, and allows export to non-Python environments.

```python
import tensorflow as tf
import numpy as np

print(f"TensorFlow: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")  # True by default

# Eager: immediate execution
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.matmul(a, b)
print(c)           # EagerTensor with values shown
print(c.numpy())   # convert to numpy array

# Disable eager (not recommended, for legacy code)
# tf.compat.v1.disable_eager_execution()

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # avoid full GPU allocation at start
print("GPUs available:", len(gpus))
```

---

## 2. Tensors: tf.Tensor, tf.Variable, tf.constant

### 2.1 Creating Tensors

```python
# tf.constant: immutable tensor
t_scalar = tf.constant(3.14)                             # 0-D scalar
t_vector = tf.constant([1, 2, 3, 4])                    # 1-D
t_matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])        # 2-D
t_3d     = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3-D

# dtype specified explicitly
t_f16  = tf.constant([1.0, 2.0], dtype=tf.float16)
t_int  = tf.constant([1, 2, 3], dtype=tf.int32)
t_bool = tf.constant([True, False, True])
t_str  = tf.constant(["hello", "world"])  # string tensor

# From NumPy
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t   = tf.constant(arr)

# Factory functions
zeros    = tf.zeros([3, 4], dtype=tf.float32)
ones     = tf.ones([3, 4])
full     = tf.fill([3, 4], value=7.0)
eye      = tf.eye(4)
zeros_like = tf.zeros_like(t_matrix)
ones_like  = tf.ones_like(t_matrix)

# Random tensors
normal  = tf.random.normal([3, 4], mean=0.0, stddev=1.0)
uniform = tf.random.uniform([3, 4], minval=0.0, maxval=1.0)
trunc   = tf.random.truncated_normal([3, 4], mean=0.0, stddev=1.0)  # clip > 2σ
ints    = tf.random.uniform([3, 4], minval=0, maxval=10, dtype=tf.int32)
shuffled= tf.random.shuffle(tf.constant([1, 2, 3, 4, 5]))

# Range
arange  = tf.range(0, 10, delta=2)         # [0, 2, 4, 6, 8]
linspace= tf.linspace(0.0, 1.0, num=11)
```

### 2.2 Tensor Attributes and Operations

```python
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Attributes
print(x.shape)     # TensorShape([2, 3])
print(x.dtype)     # tf.float32
print(x.ndim)      # 2
print(x.device)    # /job:localhost/replica:0/task:0/device:CPU:0

# Convert
print(x.numpy())           # numpy array
print(x[0].numpy())        # first row

# Casting
x_int  = tf.cast(x, tf.int32)
x_f64  = tf.cast(x, tf.float64)
x_f16  = tf.cast(x, tf.float16)

# Shape operations
print(tf.shape(x))           # dynamic shape (useful in tf.function)
print(x.shape.as_list())     # [2, 3]
reshaped  = tf.reshape(x, [3, 2])
transposed= tf.transpose(x)          # (3, 2)
expanded  = tf.expand_dims(x, axis=0) # (1, 2, 3)
squeezed  = tf.squeeze(expanded)       # (2, 3)
flattened = tf.reshape(x, [-1])        # (6,)

# Concat and stack
a = tf.ones([2, 3])
b = tf.ones([2, 3])
cat_0 = tf.concat([a, b], axis=0)    # (4, 3)
cat_1 = tf.concat([a, b], axis=1)    # (2, 6)
stacked = tf.stack([a, b], axis=0)   # (2, 2, 3)

# Split
parts = tf.split(cat_0, num_or_size_splits=2, axis=0)  # list of 2 (2,3) tensors

# Arithmetic
c = tf.add(a, b)         # or a + b
c = tf.subtract(a, b)    # or a - b
c = tf.multiply(a, b)    # or a * b  (element-wise)
c = tf.divide(a, b)      # or a / b

# Matrix ops
m1 = tf.random.normal([3, 4])
m2 = tf.random.normal([4, 5])
mm = tf.matmul(m1, m2)            # (3, 5)
mm = m1 @ m2                      # same with @ operator

# Batched matmul
bm1 = tf.random.normal([10, 3, 4])
bm2 = tf.random.normal([10, 4, 5])
bmm = tf.linalg.matmul(bm1, bm2)  # (10, 3, 5)

# Reductions
print(tf.reduce_sum(x))               # scalar sum
print(tf.reduce_sum(x, axis=0))       # (3,)
print(tf.reduce_mean(x, axis=1, keepdims=True))  # (2, 1)
print(tf.reduce_max(x))
print(tf.reduce_min(x))
print(tf.reduce_prod(x))

# Element-wise math
print(tf.math.exp(x))
print(tf.math.log(x))
print(tf.math.sqrt(x))
print(tf.math.abs(x - 3.0))
print(tf.math.pow(x, 2))

# Logical
mask = x > 3.0
print(tf.boolean_mask(x, mask))   # [4., 5., 6.]

# Gather
indices = tf.constant([0, 2])
print(tf.gather(x, indices, axis=1))  # columns 0 and 2

# Einstein summation
print(tf.einsum('ij,jk->ik', m1, m2))
```

### 2.3 tf.Variable

Variables are mutable state — parameters, running averages, etc.

```python
# Create variable
w = tf.Variable(tf.random.normal([4, 3]), name='weights', trainable=True)
b = tf.Variable(tf.zeros([3]), name='bias')

# Assignment
w.assign(tf.ones([4, 3]))         # in-place replace
w.assign_add(tf.ones([4, 3]))     # w += delta
w.assign_sub(0.01 * tf.ones([4, 3]))  # w -= delta

# Read
print(w.numpy())
print(w.shape, w.dtype, w.name)
print(w.trainable)   # True

# Scatter updates
w.scatter_update(tf.IndexedSlices(values=tf.ones([2, 3]), indices=[0, 2]))

# Non-trainable variable (e.g., step counter)
step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
step.assign_add(1)

# Variable collections
print(w.device)

# GradientTape tracks variable operations
x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
with tf.GradientTape() as tape:
    y = tf.matmul(x, w) + b         # w and b are automatically watched
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, [w, b])
print(grads[0].shape)  # same as w: (4, 3)
print(grads[1].shape)  # same as b: (3,)
```

---

## 3. tf.function: Graph Compilation, Tracing, Retracing

### 3.1 Basic tf.function

```python
# Eager function (slow for production)
def compute_eager(x, y):
    return tf.matmul(x, y)

# Graph function (compiled, fast)
@tf.function
def compute_graph(x, y):
    return tf.matmul(x, y)

a = tf.random.normal([100, 200])
b = tf.random.normal([200, 300])

# First call: TRACING — Python code runs to build graph
c = compute_graph(a, b)

# Subsequent calls: fast graph execution (no Python overhead)
for _ in range(1000):
    c = compute_graph(a, b)
```

### 3.2 Understanding Tracing and Retracing

**Retracing** occurs when input shape/dtype changes or Python control flow differs. Each trace creates a new graph; too many traces hurt performance. Use `input_signature` to constrain shapes, and `tf.cond`/`tf.while_loop` instead of Python `if`/`for` for graph-internal logic.

```python
@tf.function
def f(x):
    print("Tracing!")   # Python print — only runs during tracing
    tf.print("Running!") # TF print — runs every call
    return x + 1

f(tf.constant(1))  # prints "Tracing!" then "Running!"
f(tf.constant(2))  # prints "Running!" only (same dtype/shape → no retrace)
f(tf.constant(1.0))  # prints "Tracing!" — different dtype → retrace!

# input_signature: prevent retracing for different shapes
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def stable_fn(x):
    return x @ tf.random.normal([784, 10])

# Concrete function (compiled for specific input spec)
concrete = stable_fn.get_concrete_function()
```

### 3.3 tf.function with Control Flow

```python
@tf.function
def conditional(x, training):
    if training:                         # Python if (traced for each value!)
        return x * 2
    return x

# Use tf.cond for conditional inside graph
@tf.function
def graph_conditional(x, training):
    return tf.cond(training, lambda: x * 2, lambda: x)

# tf.while_loop for loops (alternative to Python for inside tf.function)
@tf.function
def sum_loop(n):
    i = tf.constant(0)
    total = tf.constant(0.0)
    cond  = lambda i, _: i < n
    body  = lambda i, t: (i + 1, t + tf.cast(i, tf.float32))
    _, result = tf.while_loop(cond, body, [i, total])
    return result

print(sum_loop(tf.constant(10)))  # 45.0
```

### 3.4 XLA Compilation

**XLA** (Accelerated Linear Algebra) compiles TF graphs into optimized machine code. Enable via `jit_compile=True` in `tf.function`:

```python
@tf.function(jit_compile=True)  # XLA compilation (TF 2.7+)
def xla_fast(x):
    return tf.linalg.matmul(x, tf.random.normal([784, 10]))

# Or globally
tf.config.optimizer.set_jit(True)
```

**Benefits**: Fused ops, better memory layout, TPU-native. **Gotchas**: Fixed shapes preferred; some ops unsupported; first run slower (compile). Use for inference or stable training loops.

### 3.5 GradientTape

```python
x = tf.Variable(3.0)

# Basic gradient
with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1

dy_dx = tape.gradient(y, x)   # 2x + 2 = 8
print(dy_dx)

# Gradient w.r.t. non-Variable (watch it)
x_const = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x_const)
    y = x_const ** 3

dy = tape.gradient(y, x_const)   # 3x^2 = 27
print(dy)

# Multiple outputs
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y1 = x ** 2
    y2 = x ** 3

grads = tape.gradient([y1, y2], x)  # gradient of sum: 2x + 3x^2 = 16
print(grads)

# Persistent tape (for multiple gradient calls)
with tf.GradientTape(persistent=True) as tape:
    y = x ** 4

dy   = tape.gradient(y, x)    # 4x^3
d2y  = tape.gradient(dy, x)   # 12x^2
del tape  # manually release

# Higher-order gradients
x = tf.Variable(2.0)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x ** 3
    dy = t1.gradient(y, x)      # 3x^2
d2y = t2.gradient(dy, x)       # 6x
print(d2y)  # 12.0
```

---

## 4. Keras API: Sequential, Functional, Subclassing

### 4.1 Sequential API

Best for simple, linear stacks of layers.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax'),
], name='mlp')

model.summary()

# Add layers incrementally
model2 = keras.Sequential(name='mlp2')
model2.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model2.add(layers.Dense(10))
```

### 4.2 Functional API

Supports multiple inputs/outputs, shared layers, and non-sequential topologies.

```python
# Single input/output
inputs  = keras.Input(shape=(784,), name='pixels')
x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='functional_mlp')

# Multiple inputs
input_a = keras.Input(shape=(64,), name='input_a')
input_b = keras.Input(shape=(32,), name='input_b')
x_a = layers.Dense(64, activation='relu')(input_a)
x_b = layers.Dense(32, activation='relu')(input_b)
merged  = layers.Concatenate()([x_a, x_b])
out_cls = layers.Dense(5, activation='softmax', name='class')(merged)
out_reg = layers.Dense(1, name='value')(merged)

multi_model = keras.Model(inputs=[input_a, input_b], outputs=[out_cls, out_reg])

# Residual block as functional
def residual_block(x, filters):
    h = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(filters, 3, padding='same', use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    # If shape mismatch, project shortcut
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, use_bias=False)(x)
    return layers.Add()([x, h])

inp = keras.Input(shape=(32, 32, 3))
x   = layers.Conv2D(64, 3, padding='same')(inp)
x   = residual_block(x, 64)
x   = residual_block(x, 64)
x   = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(10, activation='softmax')(x)
resnet_like = keras.Model(inp, out)
```

### 4.3 Model Subclassing

Maximum flexibility — pure Python class.

```python
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn    = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn     = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(d_model),
        ])
        self.ln1     = layers.LayerNormalization(epsilon=1e-6)
        self.ln2     = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_out = self.attn(x, x, attention_mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.ln1(x + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.ln2(out1 + ffn_out)


class TextTransformer(keras.Model):
    def __init__(self, vocab_size, max_len, d_model, num_heads, ff_dim, num_blocks, num_classes, rate=0.1):
        super().__init__()
        self.embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = layers.Embedding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, ff_dim, rate) for _ in range(num_blocks)]
        self.pool   = layers.GlobalAveragePooling1D()
        self.drop   = layers.Dropout(rate)
        self.clf    = layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        pos = tf.range(seq_len)[tf.newaxis, :]    # (1, seq_len)
        x   = self.embed(x) + self.pos_embed(pos) # (B, L, d_model)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.pool(x)
        x = self.drop(x, training=training)
        return self.clf(x)

model = TextTransformer(vocab_size=20000, max_len=256, d_model=128,
                        num_heads=4, ff_dim=512, num_blocks=4, num_classes=5)
model.build(input_shape=(None, 256))
model.summary()
```

---

## 5. Common Layers

### 5.1 Dense and Activation

```python
# Dense: output = activation(input @ kernel + bias)
dense = layers.Dense(
    units=256,
    activation='relu',       # or tf.nn.relu, keras.activations.relu
    use_bias=True,
    kernel_initializer='glorot_uniform',  # Xavier
    bias_initializer='zeros',
    kernel_regularizer=keras.regularizers.l2(1e-4),
    bias_regularizer=None,
    activity_regularizer=None,
)

# Activation layers (standalone)
relu    = layers.ReLU()
leaky   = layers.LeakyReLU(alpha=0.2)
prelu   = layers.PReLU()           # learnable alpha
elu     = layers.ELU(alpha=1.0)
gelu    = layers.Activation('gelu')
silu    = layers.Activation('swish')  # also called SiLU
softmax = layers.Softmax(axis=-1)
sigmoid = layers.Activation('sigmoid')
tanh    = layers.Activation('tanh')
```

### 5.2 Convolutional Layers

```python
# Conv2D: (batch, height, width, channels) → (batch, H', W', filters)
# H' = floor((H + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
conv2d = layers.Conv2D(
    filters=64, kernel_size=(3, 3), strides=(1, 1),
    padding='same',             # 'valid' or 'same'
    dilation_rate=(1, 1),
    groups=1,                   # groups=in_channels → depthwise
    use_bias=False,             # usually False with BN
    kernel_initializer='he_normal',
    activation=None,
)

# Depthwise separable
dw_sep = layers.SeparableConv2D(128, kernel_size=3, padding='same')

# Depthwise only
dw     = layers.DepthwiseConv2D(kernel_size=3, padding='same')

# Transpose conv (upsampling / decoder)
conv_t = layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='same')

# 1D for sequences
conv1d = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')

# Pooling
max_pool  = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
avg_pool  = layers.AveragePooling2D(pool_size=(2, 2))
gap       = layers.GlobalAveragePooling2D()
gmp       = layers.GlobalMaxPooling2D()
gap1d     = layers.GlobalAveragePooling1D()

# Example CNN block
def cnn_block(filters, x):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x
```

### 5.3 Recurrent Layers

```python
# LSTM
lstm = layers.LSTM(
    units=256,
    return_sequences=True,   # return all timestep outputs (vs only last)
    return_state=True,        # also return final (h, c)
    dropout=0.3,              # input dropout
    recurrent_dropout=0.0,    # recurrent dropout (slow, use carefully)
    go_backwards=False,
    stateful=False,
    unroll=False,
)

x = tf.random.normal([16, 50, 128])  # (batch, timesteps, features)
output, final_h, final_c = lstm(x)
print(output.shape)   # (16, 50, 256)
print(final_h.shape)  # (16, 256)

# Bidirectional wrapper
bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
out = bilstm(x)
print(out.shape)  # (16, 50, 256)

# GRU
gru = layers.GRU(128, return_sequences=True, return_state=True)
out, h = gru(x)

# SimpleRNN (rarely used in practice)
rnn = layers.SimpleRNN(64, return_sequences=True)

# Stacked RNN
stacked = keras.Sequential([
    layers.LSTM(256, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64),
    layers.Dense(10, activation='softmax'),
])
```

### 5.4 Normalization Layers

```python
# BatchNormalization
# Normalizes per feature (channel) over batch + spatial dims
# μ, σ computed per feature over batch during training
# During inference: uses running mean/var (EMA)
bn = layers.BatchNormalization(
    axis=-1,           # feature axis (default: last)
    momentum=0.99,     # EMA momentum for running stats
    epsilon=1e-3,
    center=True,       # add β (bias)
    scale=True,        # multiply by γ (weight)
)

# LayerNormalization
# Normalizes per sample over last D dimensions
# Used in Transformers (stable, batch-size independent)
ln = layers.LayerNormalization(axis=-1, epsilon=1e-6)

# GroupNormalization (TF addon or TF 2.16+)
# Divides channels into groups, normalizes within each
# Good for small batches (detection, segmentation)
# from tensorflow_addons.layers import GroupNormalization
# gn = GroupNormalization(groups=8)

# InstanceNormalization (style transfer, GAN)
# from tensorflow_addons.layers import InstanceNormalization
# inst = InstanceNormalization(axis=-1)
```

### 5.5 Attention Layers

```python
# Multi-head attention (Keras 2.4+)
mha = layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,          # d_k per head
    value_dim=None,      # defaults to key_dim
    dropout=0.1,
    use_bias=True,
    attention_axes=None,
)

q = k = v = tf.random.normal([4, 20, 512])
out, weights = mha(q, k, v, return_attention_scores=True)
print(out.shape)      # (4, 20, 512)
print(weights.shape)  # (4, 8, 20, 20)  attention weights per head

# Cross-attention (q from one sequence, k/v from another)
q_seq = tf.random.normal([4, 15, 512])
kv_seq= tf.random.normal([4, 20, 512])
out = mha(q_seq, kv_seq, kv_seq)
print(out.shape)  # (4, 15, 512)

# Causal mask for decoder
def causal_mask(seq_len):
    """Upper triangular mask — attend only to past positions."""
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask[tf.newaxis, tf.newaxis], tf.bool)

# Embedding + Positional Encoding
embed = layers.Embedding(input_dim=10000, output_dim=256, mask_zero=True)
pos   = layers.Embedding(input_dim=512, output_dim=256)

tokens = tf.random.uniform([8, 50], 0, 10000, dtype=tf.int32)
positions = tf.range(50)[tf.newaxis, :]
embedded  = embed(tokens) + pos(positions)
print(embedded.shape)  # (8, 50, 256)
```

---

## 6. Custom Components

### 6.1 Custom Layer

```python
class ReZeroLayer(keras.layers.Layer):
    """
    ReZero: residual connection with learned scalar α.
    Forward: x + α * F(x), α initialized to 0.
    """
    def __init__(self, sub_layer, **kwargs):
        super().__init__(**kwargs)
        self.sub_layer = sub_layer

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha', shape=(), initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, x, training=False):
        return x + self.alpha * self.sub_layer(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({'sub_layer': keras.layers.serialize(self.sub_layer)})
        return config


class GatedLinearUnit(keras.layers.Layer):
    """GLU: x → sigmoid(W1 x) ⊙ (W2 x)"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.proj = layers.Dense(self.units * 2)

    def call(self, x):
        x = self.proj(x)
        x, gate = tf.split(x, 2, axis=-1)
        return x * tf.sigmoid(gate)

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config


class SpatialAttention(keras.layers.Layer):
    """Channel-wise and spatial attention (CBAM-style)."""
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        C = input_shape[-1]
        self.gap  = layers.GlobalAveragePooling2D()
        self.gmp  = layers.GlobalMaxPooling2D()
        self.fc1  = layers.Dense(C // self.ratio, activation='relu')
        self.fc2  = layers.Dense(C)
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x, training=False):
        # Channel attention
        avg_pool = self.fc2(self.fc1(self.gap(x)))  # (B, C)
        max_pool = self.fc2(self.fc1(self.gmp(x)))  # (B, C)
        ch_attn  = tf.sigmoid(avg_pool + max_pool)[:, tf.newaxis, tf.newaxis, :]
        x = x * ch_attn

        # Spatial attention
        avg_sp = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_sp = tf.reduce_max(x, axis=-1, keepdims=True)
        sp_attn= self.conv(tf.concat([avg_sp, max_sp], axis=-1))
        return x * sp_attn
```

### 6.2 Custom Model (Variational Autoencoder)

```python
class Sampling(keras.layers.Layer):
    """Reparameterization trick: z = μ + ε·σ, ε ~ N(0,1)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAE(keras.Model):
    def __init__(self, latent_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.total_loss = keras.metrics.Mean(name='total_loss')
        self.recon_loss = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss    = keras.metrics.Mean(name='kl_loss')

        # Encoder
        self.encoder = keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        self.z_mean    = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.sampling  = Sampling()

        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28, 1)),
        ])

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, training=False):
        z_mean, z_log_var = self.encode(x)
        z = self.sampling([z_mean, z_log_var])
        return self.decode(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decode(z)

            # Reconstruction loss (per pixel binary CE)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            # KL divergence: -0.5 * Σ(1 + log σ² - μ² - σ²)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total = recon_loss + kl_loss

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss.update_state(total)
        self.recon_loss.update_state(recon_loss)
        self.kl_loss.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.total_loss, self.recon_loss, self.kl_loss]
```

### 6.3 Custom Loss Functions

```python
# Function-style loss
def dice_loss(y_true, y_pred, smooth=1e-7):
    """Dice loss for segmentation."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

# Class-style loss
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss: FL(p_t) = -α_t (1-p_t)^γ log(p_t)
    Addresses class imbalance by down-weighting easy examples.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        fl = alpha_t * tf.pow(1 - p_t, self.gamma) * bce
        return tf.reduce_mean(fl)

    def get_config(self):
        base = super().get_config()
        base.update({'gamma': self.gamma, 'alpha': self.alpha})
        return base


class ContrastiveLoss(keras.losses.Loss):
    """Contrastive loss: L = y*d^2 + (1-y)*max(m-d, 0)^2"""
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        d = tf.reduce_sum(tf.square(y_pred[:, 0] - y_pred[:, 1]), axis=-1)
        d = tf.sqrt(d + 1e-9)
        y = tf.cast(y_true, tf.float32)
        loss = y * tf.square(d) + (1 - y) * tf.square(tf.maximum(self.margin - d, 0))
        return tf.reduce_mean(loss)
```

### 6.4 Custom Metrics

```python
class F1Score(keras.metrics.Metric):
    """Binary F1 score metric."""
    def __init__(self, threshold=0.5, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight('tp', initializer='zeros')
        self.fp = self.add_weight('fp', initializer='zeros')
        self.fn = self.add_weight('fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.bool)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.bool)

        tp = tf.reduce_sum(tf.cast(y_true & y_pred, tf.float32))
        fp = tf.reduce_sum(tf.cast(~y_true & y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true & ~y_pred, tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall    = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)


class TopKAccuracy(keras.metrics.Metric):
    def __init__(self, k=5, name='top_k_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.correct = self.add_weight('correct', initializer='zeros')
        self.total   = self.add_weight('total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, self.k)
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.)
        self.total.assign(0.)
```

---

## 7. Training

### 7.1 model.compile

```python
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-2,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        clipnorm=1.0,         # gradient clipping by norm
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        'accuracy',
        keras.metrics.SparseCategoricalAccuracy(name='acc'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
    ],
    # For multiple outputs:
    # loss={'class': 'ce', 'bbox': 'mse'},
    # loss_weights={'class': 1.0, 'bbox': 0.5},
)
```

### 7.2 model.fit

```python
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_val, y_val),
    # validation_split=0.1,   # alternative to validation_data
    callbacks=[...],
    class_weight={0: 1.0, 1: 3.0},  # handle class imbalance
    sample_weight=sample_weights,   # per-sample weight
    initial_epoch=0,                # for resuming
    steps_per_epoch=None,           # auto from data length
    verbose=1,                      # 0=silent, 1=progress, 2=one line
    shuffle=True,
)

# History
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend(); plt.show()
```

### 7.3 Custom Training Loop

```python
optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
loss_fn   = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric   = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss   = loss_fn(y, logits)
        # Add regularization losses
        loss  += tf.add_n(model.losses) if model.losses else 0.0
    grads = tape.gradient(loss, model.trainable_weights)
    # Gradient clipping
    grads, _  = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss

@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    val_acc_metric.update_state(y, logits)

EPOCHS = 20
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch+1}/{EPOCHS}')
    train_loss_sum = 0.0

    # Training
    for step, (x_batch, y_batch) in enumerate(train_ds):
        loss = train_step(x_batch, y_batch)
        train_loss_sum += loss.numpy()
        if step % 100 == 0:
            print(f'  step {step}: loss={loss:.4f}')

    train_acc = train_acc_metric.result().numpy()
    train_acc_metric.reset_state()

    # Validation
    for x_batch, y_batch in val_ds:
        val_step(x_batch, y_batch)
    val_acc = val_acc_metric.result().numpy()
    val_acc_metric.reset_state()

    print(f'  Train acc={train_acc:.4f}  Val acc={val_acc:.4f}')
```

---

## 8. Callbacks

### 8.1 Built-in Callbacks

```python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=1e-4,
    mode='min',
    restore_best_weights=True,  # revert to best epoch
    start_from_epoch=5,
    verbose=1
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}_{val_acc:.4f}.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    save_freq='epoch',
    verbose=1,
)

# TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,       # log weight histograms every N epochs
    write_graph=True,
    write_images=True,
    write_steps_per_second=True,
    update_freq='epoch',    # or integer for steps
    profile_batch='500,520' # profile batches 500-520
)

# Reduce LR on plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_delta=1e-4,
    min_lr=1e-7,
    cooldown=2,
    verbose=1,
)

# LR scheduling
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    return lr * 0.95

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# CSV logger
csv_logger = keras.callbacks.CSVLogger('training_log.csv', separator=',', append=False)

# Terminate on NaN
terminate_nan = keras.callbacks.TerminateOnNaN()

# Lambda (quick one-off callback)
lambda_cb = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"LR: {model.optimizer.lr.numpy():.6f}")
)

callbacks = [early_stop, checkpoint, tensorboard, reduce_lr, csv_logger, terminate_nan]
```

### 8.2 Custom Callback

```python
class GradientMonitor(keras.callbacks.Callback):
    """Log gradient norms every N batches."""
    def __init__(self, log_freq=100, tb_writer=None):
        super().__init__()
        self.log_freq  = log_freq
        self.tb_writer = tb_writer
        self._step     = 0

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_freq != 0:
            return
        # Compute gradient norms
        with tf.GradientTape() as tape:
            x, y = self._get_sample()
            loss = self.model.compiled_loss(y, self.model(x, training=False))
        grads = tape.gradient(loss, self.model.trainable_weights)
        norms = [tf.norm(g).numpy() for g in grads if g is not None]
        avg_norm = np.mean(norms)
        if self.tb_writer:
            with self.tb_writer.as_default():
                tf.summary.scalar('grad_norm', avg_norm, step=self._step)
        self._step += 1


class WarmupCosineSchedule(keras.callbacks.Callback):
    """Linear warmup + cosine decay learning rate schedule."""
    def __init__(self, warmup_epochs, total_epochs, max_lr, min_lr=1e-7):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        import math
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f'\nEpoch {epoch+1}: LR={lr:.2e}')


class WeightedEnsemble(keras.callbacks.Callback):
    """Save model weights at each epoch for SWA (Stochastic Weight Averaging)."""
    def __init__(self, avg_start=10):
        super().__init__()
        self.avg_start = avg_start
        self._avg_weights = None
        self._count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.avg_start:
            return
        self._count += 1
        current = self.model.get_weights()
        if self._avg_weights is None:
            self._avg_weights = current
        else:
            self._avg_weights = [
                (a * (self._count - 1) + c) / self._count
                for a, c in zip(self._avg_weights, current)
            ]

    def on_train_end(self, logs=None):
        if self._avg_weights:
            self.model.set_weights(self._avg_weights)
            print('Applied SWA weights.')
```

---

## 9. tf.data: Efficient Data Pipelines

### 9.1 Creating Datasets

```python
# From tensors / NumPy
ds_np = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# From Python generator (memory efficient for large data)
def image_generator():
    for path, label in zip(image_paths, labels):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        yield img, label

ds_gen = tf.data.Dataset.from_generator(
    image_generator,
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
)

# From TFRecord files
def parse_tfrecord(serialized):
    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized, feature_desc)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) / 255.0
    return image, example['label']

ds_tfr = tf.data.TFRecordDataset(['data.tfrecord']).map(parse_tfrecord)

# From directory (images)
ds_dir = keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='int',  # or 'categorical', 'binary'
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training',
)
```

### 9.2 Transformations

```python
# Augmentation pipeline
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    # Random crop
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, [224, 224])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def normalize(image, label):
    mean = tf.constant([0.485, 0.456, 0.406])
    std  = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (ds_np
    .cache()                                # cache after first epoch
    .shuffle(buffer_size=10000, seed=42)    # full random shuffle
    .map(augment, num_parallel_calls=AUTOTUNE)   # parallel preprocessing
    .map(normalize, num_parallel_calls=AUTOTUNE)
    .batch(64, drop_remainder=True)
    .prefetch(AUTOTUNE)                     # overlap data prep with model
)

val_ds = (val_dataset
    .cache()
    .map(normalize, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

# Filter
ds_filtered = ds_np.filter(lambda x, y: y != 0)

# Repeat
ds_repeated = ds_np.repeat(3)   # or .repeat() for infinite

# Interleave (merge multiple dataset files)
files = tf.data.Dataset.list_files('data/shard_*.tfrecord')
ds_interleaved = files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=8,
    block_length=1,
    num_parallel_calls=AUTOTUNE,
    deterministic=False,
)

# Zip
label_ds = tf.data.Dataset.from_tensor_slices(labels)
image_ds = tf.data.Dataset.from_tensor_slices(images)
zipped   = tf.data.Dataset.zip((image_ds, label_ds))

# Window (for time series)
ts_ds = tf.data.Dataset.range(1000)
windowed = ts_ds.window(size=32, shift=1, drop_remainder=True)
windowed = windowed.flat_map(lambda w: w.batch(32))
```

### 9.3 Performance Tips

```python
# Benchmark dataset performance
import time

def benchmark(dataset, num_epochs=2):
    start = time.perf_counter()
    for epoch in range(num_epochs):
        for batch in dataset:
            pass
    return time.perf_counter() - start

# Optimal pipeline:
optimal_ds = (
    tf.data.Dataset.from_tensor_slices(data)
    .cache()          # 1. Cache raw data
    .shuffle(10000)   # 2. Shuffle
    .map(preprocess, num_parallel_calls=AUTOTUNE)  # 3. Preprocess in parallel
    .batch(batch_size, drop_remainder=True)          # 4. Batch
    .prefetch(AUTOTUNE)  # 5. Prefetch next batch while model trains
)
```

---

## 10. Distributed Training

### 10.1 MirroredStrategy (Single Machine, Multi-GPU)

```python
strategy = tf.distribute.MirroredStrategy()
print(f"Replicas: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # All variables created here are mirrored across GPUs
    model = keras.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(32,32,3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01 * strategy.num_replicas_in_sync),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Scale batch size
GLOBAL_BATCH = 128 * strategy.num_replicas_in_sync
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(GLOBAL_BATCH)

model.fit(train_ds, epochs=10)
```

### 10.2 MultiWorkerMirroredStrategy (Multi-Machine)

```python
# Set via environment before process start:
# TF_CONFIG = {
#   "cluster": {"worker": ["host1:port", "host2:port"]},
#   "task": {"type": "worker", "index": 0}
# }
import json, os

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
    )
)

with strategy.scope():
    model = create_model()
    model.compile(...)

model.fit(global_train_ds, epochs=10)
```

### 10.3 TPUStrategy

```python
# Google Cloud TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

print(f"All TPU devices: {tf.config.list_logical_devices('TPU')}")

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# TPU needs fixed batch size and shapes
# Batch size must be divisible by 8 (TPU cores)
model.fit(tpu_ds, epochs=5)
```

### 10.4 Custom Distribution Loop

```python
# For fine-grained control
strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = 128

with strategy.scope():
    model     = create_model()
    optimizer = keras.optimizers.Adam()
    loss_obj  = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE  # no reduction for manual
    )

def compute_loss(y_true, y_pred):
    per_example_loss = loss_obj(y_true, y_pred)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function
def distributed_train_step(dist_inputs):
    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = compute_loss(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    per_replica_loss = strategy.run(step_fn, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
```

---

## 11. Mixed Precision

```python
# Enable mixed precision globally
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Compute dtype: {policy.compute_dtype}")    # float16
print(f"Variable dtype: {policy.variable_dtype}")  # float32

# Build model — Keras handles precision automatically
# Only make sure the output layer (softmax/sigmoid) is float32
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),  # uses float16
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32'),   # must be float32
])

# Wrap optimizer with LossScaleOptimizer for numerical stability
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# For custom training loop:
with tf.GradientTape() as tape:
    outputs = model(x_batch, training=True)
    loss    = loss_fn(y_batch, outputs)
    scaled_loss = optimizer.get_scaled_loss(loss)

scaled_grads = tape.gradient(scaled_loss, model.trainable_weights)
grads = optimizer.get_unscaled_gradients(scaled_grads)
optimizer.apply_gradients(zip(grads, model.trainable_weights))

# bfloat16 (recommended for TPU/newer GPUs)
bf16_policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(bf16_policy)
```

---

## 12. Saving and Deployment

### 12.1 SavedModel Format

```python
# Save full model (recommended)
model.save('saved_model/my_model')            # SavedModel format
model.save('my_model.keras')                  # Keras native format

# Load
loaded = keras.models.load_model('saved_model/my_model')

# Save weights only
model.save_weights('weights.h5')              # H5 format
model.save_weights('weights/ckpt')            # TF checkpoint format
model.load_weights('weights.h5')

# Checkpointing with tf.train.Checkpoint
ckpt    = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step_counter)
manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)

# Save
manager.save()

# Restore latest
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")

# Export tf.function as SavedModel for inference
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def serving_fn(x):
    return model(x, training=False)

tf.saved_model.save(model, 'export/1',
                    signatures={'serving_default': serving_fn})
```

### 12.2 TensorFlow Lite (Edge Deployment)

```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')

# Optimization: dynamic range quantization (reduces model to ~1/4 size)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full integer quantization (for hardware accelerators)
def representative_data_gen():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

converter.representative_dataset     = representative_data_gen
converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type       = tf.int8
converter.inference_output_type      = tf.int8

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Run TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], x_test[:1])
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(output.shape)
```

### 12.3 TensorFlow Serving

```bash
# Save model in serving format (version directory)
# tf.saved_model.save(model, 'serving/mnist/1')

# Run TF Serving with Docker
docker run -t --rm -p 8501:8501 \
  --mount type=bind,source=/path/to/serving,target=/models/mnist \
  -e MODEL_NAME=mnist \
  tensorflow/serving

# Make REST API predictions
curl -d '{"instances": [[0.1, 0.2, ...]]}' \
     -X POST http://localhost:8501/v1/models/mnist:predict
```

```python
import requests, json, numpy as np

data = {"instances": x_test[:5].tolist()}
response = requests.post(
    'http://localhost:8501/v1/models/mnist:predict',
    data=json.dumps(data),
    headers={"Content-Type": "application/json"}
)
predictions = response.json()['predictions']
print(np.argmax(predictions, axis=-1))

# Get model metadata
info = requests.get('http://localhost:8501/v1/models/mnist/metadata')
print(info.json())
```

---

## Common Pitfalls and Debugging

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Retracing in tf.function** | Slow first batches, many "Tracing!" logs | Use `input_signature`, avoid Python control flow with tensor-dependent branches |
| **Python `if training` inside @tf.function** | Different graph per train/eval path | Use `tf.cond(training, ...)` or separate functions |
| **Eager-only ops in graph** | Error when calling from tf.function | Replace with graph-compatible ops (e.g. `tf.print` not `print`) |
| **Shape/dtype mismatch** | Retracing or errors | Ensure consistent shapes; use `TensorSpec(shape=[None,...])` for variable batch |
| **tf.Variable in wrong scope** | Variables recreated each trace | Create variables outside `@tf.function` or in `tf.init_scope()` |
| **tf.data not prefetched** | GPU starved, low utilization | Use `.prefetch(AUTOTUNE)`, `num_parallel_calls=AUTOTUNE` |
| **Mixed precision output layer** | NaN in loss with float16 | Set output layer `dtype='float32'`; use `LossScaleOptimizer` |
| **TF Serving version mismatch** | Load errors | Match SavedModel format; use `saved_model_cli` to inspect |

---

## Production Deployment Notes

- **SavedModel**: Default export; version subdirs (`/1`, `/2`) for rollbacks. Use `signatures` for named endpoints.
- **TF Lite**: `converter.optimizations`, `representative_dataset` for INT8; test on target device.
- **TF Serving**: Docker with `MODEL_NAME`, `MODEL_BASE_PATH`; gRPC or REST.
- **Edge**: TFLite (mobile), TF.js (browser), Coral (Edge TPU).
- **TPU**: `TPUStrategy`, fixed batch size divisible by 8; use `mixed_bfloat16`.

---

## 13. TF Hub for Transfer Learning

```python
import tensorflow_hub as hub

# Image feature extraction (EfficientNet-B0)
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
    trainable=False,   # freeze pretrained weights
    input_shape=(224, 224, 3),
)

# Build fine-tuning model
model = keras.Sequential([
    feature_extractor,
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax'),
])

# Phase 1: Train head only (frozen backbone)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Phase 2: Fine-tune (unfreeze backbone with lower LR)
feature_extractor.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=20)

# BERT for text (NLP)
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder    = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                                  trainable=True)

text_input = keras.Input(shape=(), dtype=tf.string, name='text')
preprocessed = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed)
clf_output = outputs['pooled_output']     # [CLS] token embedding (768-dim)
x = layers.Dropout(0.1)(clf_output)
predictions = layers.Dense(5, activation='softmax')(x)
bert_model = keras.Model(text_input, predictions)
```

---

## 14. TF Probability Basics

```python
import tensorflow_probability as tfp
tfd = tfp.distributions

# Basic distributions
normal   = tfd.Normal(loc=0.0, scale=1.0)
bernoulli= tfd.Bernoulli(probs=0.7)
cat      = tfd.Categorical(probs=[0.1, 0.4, 0.5])
mvn      = tfd.MultivariateNormalDiag(loc=tf.zeros(3), scale_diag=tf.ones(3))

# Sample
samples = normal.sample(1000)
log_prob = normal.log_prob(0.5)

# Mixture model
gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.3, 0.4, 0.3]),
    components_distribution=tfd.Normal(loc=[-2.0, 0.0, 3.0], scale=[0.5, 1.0, 0.5])
)

# Probabilistic layer
class BayesianLinear(keras.layers.Layer):
    """Linear layer with weight uncertainty (mean-field VI)."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        n = input_shape[-1]
        self.w_mu    = self.add_weight('w_mu',    shape=(n, self.units))
        self.w_rho   = self.add_weight('w_rho',   shape=(n, self.units), initializer='constant', value=-3)
        self.b_mu    = self.add_weight('b_mu',    shape=(self.units,), initializer='zeros')
        self.b_rho   = self.add_weight('b_rho',   shape=(self.units,), initializer='constant', value=-3)

    def call(self, x, training=False):
        if training:
            w_sigma = tf.math.softplus(self.w_rho)
            b_sigma = tf.math.softplus(self.b_rho)
            w = self.w_mu + w_sigma * tf.random.normal(self.w_mu.shape)
            b = self.b_mu + b_sigma * tf.random.normal(self.b_mu.shape)
            # KL penalty
            kl = tf.reduce_sum(tfd.Normal(0., 1.).log_prob(w) - tfd.Normal(self.w_mu, w_sigma).log_prob(w))
            self.add_loss(kl / tf.cast(tf.shape(x)[0], tf.float32))
        else:
            w, b = self.w_mu, self.b_mu
        return x @ w + b
```

---

## 15. Keras Tuner

```python
import keras_tuner as kt

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))

    # Search over number of layers
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
            activation=hp.Choice(f'activation_{i}', ['relu', 'gelu', 'elu']),
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(layers.BatchNormalization())

    model.add(layers.Dense(10, activation='softmax'))

    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    optimizer = hp.Choice('optimizer', ['adam', 'adamw', 'sgd'])
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(lr)
    elif optimizer == 'adamw':
        opt = keras.optimizers.AdamW(lr, weight_decay=hp.Float('wd', 1e-5, 1e-2, sampling='log'))
    else:
        opt = keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Bayesian Optimization tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=2,    # average over 2 runs
    directory='kt_results',
    project_name='mnist_tuning',
    overwrite=True,
)

# Hyperband tuner (more efficient)
tuner_hb = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    directory='kt_hyperband',
    project_name='mnist_hb',
)

tuner.search_space_summary()
tuner.search(
    x_train, y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best params: {best_hps.values}")

best_model = tuner.get_best_models(num_models=1)[0]
best_model.evaluate(x_test, y_test)
```

---

## 16. Full Examples

### 16.1 Image Classification Pipeline (CIFAR-10)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])
x_train = (x_train - MEAN) / STD
x_test  = (x_test - MEAN) / STD

AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.pad(image, [[4,4],[4,4],[0,0]])
    image = tf.image.random_crop(image, size=[32,32,3])
    return image, label

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(50000, seed=42)
            .map(augment_fn, num_parallel_calls=AUTOTUNE)
            .batch(128, drop_remainder=True)
            .prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .batch(256)
           .prefetch(AUTOTUNE))

# Model (ResNet-like with functional API)
def conv_bn_relu(filters, kernel_size, strides=1):
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
    ])

def res_block(x, filters, stride=1):
    shortcut = x
    h = layers.Conv2D(filters, 3, stride, padding='same', use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(filters, 3, padding='same', use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, stride, use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([h, shortcut]))

inp = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inp)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = res_block(x, 64)
x = res_block(x, 64)
x = res_block(x, 128, stride=2)
x = res_block(x, 128)
x = res_block(x, 256, stride=2)
x = res_block(x, 256)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inp, out, name='resnet_cifar')
model.summary()

# Compile
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint('best_cifar.keras', save_best_only=True, monitor='val_accuracy'),
    keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * 0.1 if epoch in [60, 120, 160] else lr
    ),
    keras.callbacks.TensorBoard('./logs/cifar10'),
]

history = model.fit(train_ds, epochs=200, validation_data=test_ds, callbacks=callbacks)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 16.2 Text Sentiment Classification (IMDB)

```python
# Data
vocab_size  = 20000
max_len     = 256
embed_dim   = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
x_test  = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding='post')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(25000).batch(64).prefetch(AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test,  y_test)).batch(256).prefetch(AUTOTUNE)

# Transformer model
inputs = keras.Input(shape=(max_len,), dtype='int32')
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)

# Positional encoding
positions = tf.range(max_len)[tf.newaxis, :]
x += layers.Embedding(max_len, embed_dim)(positions)
x = layers.Dropout(0.1)(x)

# Transformer blocks
for _ in range(2):
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim//4, dropout=0.1)(x, x)
    x    = layers.LayerNormalization()(x + layers.Dropout(0.1)(attn))
    ffn  = layers.Dense(embed_dim * 4, activation='gelu')(x)
    ffn  = layers.Dense(embed_dim)(ffn)
    x    = layers.LayerNormalization()(x + layers.Dropout(0.1)(ffn))

x   = layers.GlobalAveragePooling1D()(x)
x   = layers.Dense(64, activation='relu')(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, out)
model.compile(optimizer=keras.optimizers.AdamW(1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=15, validation_data=test_ds,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
print(f"Test acc: {model.evaluate(test_ds, verbose=0)[1]:.4f}")
```

### 16.3 Complete Custom Training Loop with Distributed Training

```python
strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH = 256

with strategy.scope():
    model     = create_model()
    optimizer = keras.optimizers.AdamW(
        keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=10000)
    )
    loss_fn   = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    train_acc = keras.metrics.SparseCategoricalAccuracy()
    val_acc   = keras.metrics.SparseCategoricalAccuracy()

def compute_loss(labels, predictions):
    per_example = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example, global_batch_size=GLOBAL_BATCH)

@tf.function
def train_step(dist_inputs):
    def step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = compute_loss(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc.update_state(y, pred)
        return loss
    losses = strategy.run(step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

@tf.function
def val_step(dist_inputs):
    def step(inputs):
        x, y = inputs
        pred = model(x, training=False)
        val_acc.update_state(y, pred)
    strategy.run(step, args=(dist_inputs,))

dist_train = strategy.experimental_distribute_dataset(train_ds)
dist_val   = strategy.experimental_distribute_dataset(val_ds)

for epoch in range(20):
    total_loss = 0.0
    for step, batch in enumerate(dist_train):
        total_loss += train_step(batch)

    for batch in dist_val:
        val_step(batch)

    print(f"Epoch {epoch+1}: loss={total_loss:.4f}  "
          f"train_acc={train_acc.result():.4f}  "
          f"val_acc={val_acc.result():.4f}")

    train_acc.reset_state()
    val_acc.reset_state()
```

---

## Resources and Further Reading

| Resource | Link |
|---|---|
| TF Documentation | tensorflow.org/api_docs |
| Keras Documentation | keras.io |
| TF Tutorials | tensorflow.org/tutorials |
| TF Hub | tfhub.dev |
| TF Model Garden | github.com/tensorflow/models |
| TF XLA | tensorflow.org/xla |
| TF Serving | tensorflow.org/tfx/guide/serving |
| TF Lite | tensorflow.org/lite |
| TF Probability | tensorflow.org/probability |
| TF Extended (TFX) | tensorflow.org/tfx |

**Key Takeaways:**
1. Eager execution is for development; `@tf.function` is for production performance
2. Use `tf.data` pipelines — `cache → shuffle → map → batch → prefetch`
3. Three model APIs: Sequential (linear), Functional (graphs), Subclassing (maximum flexibility)
4. Callbacks provide hooks into every phase of training — use them heavily
5. `MirroredStrategy` for multi-GPU, `TPUStrategy` for TPUs — minimal code change
6. Mixed precision (`mixed_float16` or `mixed_bfloat16`) for 2-3x training speedup
7. Export with `SavedModel` for deployment via TF Serving or TF Lite
