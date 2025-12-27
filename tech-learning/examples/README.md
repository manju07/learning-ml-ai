# ML/DL/CNN Practical Examples

This directory contains practical code examples for Machine Learning, Deep Learning, and Convolutional Neural Networks.

## Examples

### 1. Basic Classification (`01_basic_classification.py`)
- Simple neural network for binary classification
- Demonstrates data preprocessing, model building, and evaluation
- Includes visualization of training history

**Run:**
```bash
python 01_basic_classification.py
```

### 2. MNIST CNN (`02_mnist_cnn.py`)
- Convolutional Neural Network for handwritten digit recognition
- Demonstrates CNN architecture, batch normalization, and dropout
- Includes prediction visualization

**Run:**
```bash
python 02_mnist_cnn.py
```

### 3. Transfer Learning (`03_transfer_learning.py`)
- Using pre-trained models (VGG16, ResNet50, MobileNetV2)
- Feature extraction and fine-tuning
- Custom dataset classification

**Run:**
```bash
# Prepare your dataset first:
# - Create 'data/train' directory
# - Organize images into subdirectories by class
python 03_transfer_learning.py
```

### 4. Image Segmentation U-Net (`04_image_segmentation_unet.py`)
- U-Net architecture for semantic segmentation
- Dice loss and IoU metrics
- Medical image segmentation example

**Run:**
```bash
# Prepare your dataset:
# - Create 'data/train/images' and 'data/train/masks' directories
python 04_image_segmentation_unet.py
```

### 5. Text Classification LSTM (`05_text_classification_lstm.py`)
- LSTM for text classification
- Sentiment analysis example
- Text preprocessing and tokenization

**Run:**
```bash
python 05_text_classification_lstm.py
```

### 6. TensorFlow Basics (`06_tensorflow_basics.py`)
- TensorFlow fundamentals and tensor operations
- Building models with Sequential API
- Custom training loops
- tf.data API usage
- Model saving and loading
- Callbacks and monitoring

**Run:**
```bash
python 06_tensorflow_basics.py
```

### 7. Agentic AI with LangChain (`07_agentic_ai_langchain.py`)
- LangChain agent setup and configuration
- Tool integration (Search, Calculator, Wikipedia)
- Agents with memory and conversation
- Code generation agents
- Research agents
- Multi-tool agent examples

**Setup:**
```bash
pip install langchain langchain-openai langchain-community
pip install langchain-experimental duckduckgo-search wikipedia
export OPENAI_API_KEY="your-api-key"
```

**Run:**
```bash
python 07_agentic_ai_langchain.py
```

## Requirements

Install required packages:

```bash
pip install tensorflow numpy scikit-learn matplotlib pandas
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### For Image Classification:
1. Create directory structure:
```
data/
  train/
    class1/
      image1.jpg
      image2.jpg
    class2/
      image1.jpg
      ...
```

### For Image Segmentation:
1. Create directory structure:
```
data/
  train/
    images/
      image1.jpg
      image2.jpg
    masks/
      mask1.png
      mask2.png
```

## Tips

1. **Start Simple**: Begin with basic examples before moving to complex architectures
2. **Use GPU**: For faster training, use GPU-enabled TensorFlow
3. **Monitor Training**: Use TensorBoard to visualize training progress
4. **Data Augmentation**: Always use data augmentation for small datasets
5. **Transfer Learning**: Leverage pre-trained models for better performance

## Troubleshooting

### Out of Memory Errors:
- Reduce batch size
- Use smaller image dimensions
- Use mixed precision training

### Slow Training:
- Enable GPU acceleration
- Use data generators efficiently
- Reduce model complexity

### Poor Performance:
- Check data quality and preprocessing
- Try data augmentation
- Use transfer learning
- Tune hyperparameters

## Next Steps

1. Experiment with different architectures
2. Try different optimizers and learning rates
3. Implement custom loss functions
4. Deploy models using TensorFlow Serving or Flask/FastAPI
5. Explore advanced topics like attention mechanisms and transformers

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Papers with Code](https://paperswithcode.com/)

