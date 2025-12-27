# PyTorch: Complete Guide with Examples

## Table of Contents
1. [Introduction to PyTorch](#introduction)
2. [Tensors and Operations](#tensors)
3. [Automatic Differentiation](#autograd)
4. [Building Neural Networks](#neural-networks)
5. [Training Models](#training)
6. [Data Loading](#data-loading)
7. [Transfer Learning](#transfer-learning)
8. [Custom Modules](#custom-modules)
9. [Distributed Training](#distributed)
10. [Model Deployment](#deployment)
11. [Practical Examples](#examples)

---

## Introduction to PyTorch {#introduction}

PyTorch is a deep learning framework developed by Facebook. It's known for its dynamic computation graphs and Pythonic API.

### Key Features
- **Dynamic Computation Graphs**: Build graphs on-the-fly
- **Pythonic**: Natural Python integration
- **GPU Acceleration**: CUDA support
- **Research-Friendly**: Easy to experiment
- **Production Ready**: TorchScript, TorchServe

### Installation

```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Tensors and Operations {#tensors}

### Creating Tensors

```python
import torch
import numpy as np

# From Python lists
t1 = torch.tensor([1, 2, 3, 4])
print(f"Tensor: {t1}")

# From NumPy
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)
print(f"From NumPy: {t2}")

# Zeros and ones
t3 = torch.zeros(3, 3)
t4 = torch.ones(2, 4)
print(f"Zeros:\n{t3}")
print(f"Ones:\n{t4}")

# Random tensors
t5 = torch.randn(2, 3)  # Normal distribution
t6 = torch.rand(2, 3)   # Uniform [0, 1]
print(f"Random normal:\n{t5}")

# Range
t7 = torch.arange(0, 10, 2)
print(f"Range: {t7}")

# GPU tensors
if torch.cuda.is_available():
    t_gpu = torch.randn(3, 3).cuda()
    print(f"GPU tensor: {t_gpu.device}")
```

### Tensor Operations

```python
# Basic operations
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Element-wise
add = a + b
multiply = a * b
divide = a / b

# Matrix multiplication
matmul = torch.matmul(a, b)
# or
matmul = a @ b

# Reshaping
x = torch.randn(2, 3)
x_reshaped = x.view(3, 2)
x_flattened = x.view(-1)  # Flatten

# Transpose
x_t = x.t()

# Concatenation
cat = torch.cat([a, b], dim=0)  # Vertical
cat = torch.cat([a, b], dim=1)  # Horizontal
```

---

## Automatic Differentiation {#autograd}

### Autograd Basics

```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"dy/dx at x=2: {x.grad}")  # Should be 4

# Multiple variables
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"Gradients: {x.grad}")  # [2, 4]
```

### Custom Functions

```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return 2 * input * grad_output

# Use custom function
x = torch.tensor(3.0, requires_grad=True)
y = CustomFunction.apply(x)
y.backward()
print(f"Gradient: {x.grad}")
```

---

## Building Neural Networks {#neural-networks}

### Using nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model
model = SimpleNet(input_size=784, hidden_size=128, num_classes=10)
print(model)
```

### Sequential Models

```python
# Using Sequential
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)
```

### Common Layers

```python
# Convolutional
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Recurrent
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
gru = nn.GRU(input_size=100, hidden_size=128, batch_first=True)

# Normalization
batch_norm = nn.BatchNorm2d(64)
layer_norm = nn.LayerNorm(128)

# Dropout
dropout = nn.Dropout(0.5)
```

---

## Training Models {#training}

### Basic Training Loop

```python
import torch.optim as optim

# Model, loss, optimizer
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0
```

### Validation

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
```

---

## Data Loading {#data-loading}

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Create dataset and dataloader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Built-in Datasets

```python
from torchvision import datasets, transforms

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

---

## Transfer Learning {#transfer-learning}

### Using Pre-trained Models

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes

# Only train classifier
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Fine-tuning

```python
# Unfreeze last few layers
for param in list(model.parameters())[-10:]:
    param.requires_grad = True

# Use lower learning rate
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': list(model.parameters())[-10:], 'lr': 0.0001}
])
```

---

## Custom Modules {#custom-modules}

### Custom Layer

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

### Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

---

## Distributed Training {#distributed}

### DataParallel

```python
# Multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.cuda()
```

### DistributedDataParallel

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
```

---

## Model Deployment {#deployment}

### Saving and Loading

```python
# Save entire model
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Save state dict (recommended)
torch.save(model.state_dict(), 'model_state.pth')
model.load_state_dict(torch.load('model_state.pth'))

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### TorchScript

```python
# Convert to TorchScript
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)
traced_model.save('model_traced.pt')

# Load and use
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example)
```

---

## Practical Examples {#examples}

### Example: Image Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## Best Practices

1. **Use GPU**: `.cuda()` or `.to(device)`
2. **Set Seeds**: `torch.manual_seed(42)`
3. **Use DataLoader**: Efficient data loading
4. **Zero Gradients**: Always call `optimizer.zero_grad()`
5. **Eval Mode**: Use `model.eval()` for inference
6. **Save State Dict**: Not entire model
7. **Use nn.Module**: For custom architectures
8. **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_()`

---

## Resources

- **PyTorch Docs**: pytorch.org/docs
- **Tutorials**: pytorch.org/tutorials
- **Examples**: github.com/pytorch/examples
- **Hub**: pytorch.org/hub

---

## Conclusion

PyTorch provides a flexible and intuitive framework for deep learning. Key takeaways:

1. **Dynamic Graphs**: Build graphs on-the-fly
2. **Pythonic**: Natural Python integration
3. **Research**: Great for experimentation
4. **Production**: TorchScript for deployment
5. **Community**: Large ecosystem

Remember: PyTorch excels at research and rapid prototyping!

