# Machine Learning, Deep Learning & CNN Learning Resources

Comprehensive documentation and practical examples for learning Machine Learning, Deep Learning, and Convolutional Neural Networks.

## 📚 Documentation

### 1. [Machine Learning Guide](./learn-ml.md)
Complete guide covering:
- Introduction to ML and types of learning
- Core concepts (features, labels, overfitting, etc.)
- Data preprocessing techniques
- Common algorithms with examples (Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, K-Means, Neural Networks)
- Model building workflow
- Evaluation metrics
- Deployment strategies (Flask, FastAPI, Docker, AWS SageMaker, TensorFlow Serving, MLflow)
- End-to-end examples
- Best practices

### 2. [Deep Learning Guide](./learn-deep-learning.md)
Comprehensive deep learning documentation:
- Neural networks fundamentals
- Activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish)
- Backpropagation algorithm
- Optimization algorithms (SGD, Momentum, Adam, RMSprop)
- Regularization techniques (Dropout, L1/L2, Batch Normalization, Early Stopping)
- Neural network architectures (Feedforward, Autoencoder, VAE)
- Transfer learning
- Practical examples (MNIST, Text Classification)
- Tools and frameworks

### 3. [CNN Guide](./learn-cnn.md)
Complete Convolutional Neural Networks guide:
- CNN fundamentals and convolution operations
- Convolutional and pooling layers
- Popular architectures (LeNet-5, AlexNet, VGG, ResNet, Inception)
- Transfer learning with CNNs
- Object detection (YOLO)
- Image segmentation (U-Net, FCN)
- Practical examples (CIFAR-10, Custom datasets)
- Advanced topics (Attention mechanisms, Custom loss functions)

### 4. [TensorFlow Guide](./learn-tensorflow.md)
Comprehensive TensorFlow framework guide:
- TensorFlow fundamentals and installation
- Tensors and operations
- Building models (Sequential, Functional, Subclassing APIs)
- Training and evaluation
- Data loading with tf.data API
- Custom layers, models, and loss functions
- Callbacks and monitoring
- Distributed training (Multi-GPU, TPU)
- TensorBoard visualization
- Model saving and loading
- TensorFlow Serving for production
- Advanced topics (Mixed precision, Gradient clipping, Model pruning)
- Complete practical examples

### 5. [NLP and Transformers Guide](./learn-nlp-transformers.md)
Complete Natural Language Processing guide:
- Text preprocessing and tokenization
- Traditional NLP methods (BoW, TF-IDF, N-grams)
- Word embeddings (Word2Vec, GloVe, FastText)
- RNNs and LSTMs for NLP
- Attention mechanism and self-attention
- Transformer architecture
- BERT and variants (DistilBERT, RoBERTa, ALBERT)
- GPT and language models
- Fine-tuning pre-trained models
- Practical examples (Sentiment analysis, NER, QA, Summarization)
- Advanced topics (Transfer learning, Model compression)

### 6. [Reinforcement Learning Guide](./learn-reinforcement-learning.md)
Complete Reinforcement Learning guide:
- RL fundamentals and concepts
- Markov Decision Process (MDP)
- Value-based methods (Q-Learning, SARSA)
- Policy-based methods (Policy Gradient)
- Actor-Critic methods
- Deep Q-Networks (DQN)
- Proximal Policy Optimization (PPO)
- Practical examples (CartPole, Lunar Lander)
- Advanced topics (Prioritized Experience Replay, Double DQN)

### 7. [PyTorch Guide](./learn-pytorch.md)
Comprehensive PyTorch framework guide:
- PyTorch fundamentals and installation
- Tensors and operations
- Automatic differentiation (Autograd)
- Building neural networks (nn.Module, Sequential)
- Training models and custom training loops
- Data loading (Dataset, DataLoader)
- Transfer learning with pre-trained models
- Custom modules and layers
- Distributed training (DataParallel, DDP)
- Model deployment (TorchScript, TorchServe)
- Practical examples

### 8. [Time Series Forecasting Guide](./learn-time-series.md)
Complete Time Series Forecasting guide:
- Time series components (Trend, Seasonality, Noise)
- Preprocessing and stationarity
- Traditional methods (ARIMA, Exponential Smoothing, Prophet)
- Machine Learning methods (Linear Regression, Random Forest)
- Deep Learning methods (RNN, LSTM, GRU)
- Transformer for time series
- Multi-step forecasting
- Evaluation metrics (MAE, RMSE, MAPE)
- Practical examples (Stock prediction)

### 9. [MLOps Guide](./learn-mlops.md)
Complete MLOps (Machine Learning Operations) guide:
- MLOps lifecycle and principles
- Version control (DVC, Git LFS)
- CI/CD for ML (GitHub Actions)
- Model monitoring (Data drift, Performance monitoring)
- Model deployment (Docker, Kubernetes)
- Infrastructure as Code (Terraform)
- Experiment tracking (MLflow, Weights & Biases)
- Model registry and governance
- Best practices for production ML

### 10. [Data Science Guide](./learn-data-science.md)
Complete Data Science role and workflow guide:
- Data Science role and responsibilities
- Data Science workflow (CRISP-DM)
- Essential skills (Technical and soft skills)
- Data collection and acquisition
- Exploratory Data Analysis (EDA)
- Feature engineering techniques
- Statistical analysis methods
- Data visualization best practices
- Model development framework
- Communication and storytelling
- Tools and technologies
- Career path and progression

### 11. [Deep Python Learning Guide](./learn-python-deep.md)
Advanced Python programming guide:
- Python fundamentals review
- Advanced data structures (Collections module)
- Object-Oriented Programming (OOP) deep dive
- Functional programming concepts
- Decorators and metaclasses
- Generators and iterators
- Context managers
- Concurrency and parallelism (Threading, Multiprocessing, Async)
- Memory management
- Performance optimization (Profiling, Caching, Cython, Numba)
- Design patterns (Singleton, Factory, Observer)
- Testing and debugging
- Best practices and Pythonic code

### 12. [Agentic AI Guide](./learn-agentic-ai.md)
Complete Agentic AI frameworks and usage guide:
- Introduction to Agentic AI and autonomous agents
- Agent architecture and core components
- LangChain framework (Agents, Tools, Memory, Chains)
- LlamaIndex framework (Query engines, Agents)
- AutoGPT and AgentGPT patterns
- CrewAI framework (Multi-agent collaboration)
- Semantic Kernel (Microsoft's framework)
- Building custom agents
- Tool integration and custom tools
- Memory and state management
- Multi-agent systems and communication
- Practical examples (Research, Code generation, Data analysis)
- Best practices (Error handling, Security, Monitoring)

## 💻 Practical Examples

See the [examples](./examples/) directory for runnable code examples:

- **01_basic_classification.py**: Basic neural network for binary classification
- **02_mnist_cnn.py**: CNN for handwritten digit recognition
- **03_transfer_learning.py**: Transfer learning with pre-trained models
- **04_image_segmentation_unet.py**: U-Net for semantic segmentation
- **05_text_classification_lstm.py**: LSTM for text classification

## 🚀 Quick Start

### Installation

```bash
cd examples
pip install -r requirements.txt
```

### Run Examples

```bash
# Basic classification
python examples/01_basic_classification.py

# MNIST CNN
python examples/02_mnist_cnn.py

# Text classification
python examples/05_text_classification_lstm.py
```

## 📖 Learning Path

### Beginner
1. Master [Deep Python Learning](./learn-python-deep.md) fundamentals
2. Start with [Machine Learning Guide](./learn-ml.md)
3. Understand [Data Science](./learn-data-science.md) workflow
4. Understand core concepts and data preprocessing
5. Try basic algorithms (Linear/Logistic Regression)
6. Run `01_basic_classification.py`

### Intermediate
1. Read [Deep Learning Guide](./learn-deep-learning.md)
2. Understand neural networks and backpropagation
3. Learn about optimization and regularization
4. Run `02_mnist_cnn.py` and `05_text_classification_lstm.py`

### Advanced
1. Study [CNN Guide](./learn-cnn.md)
2. Understand advanced architectures (ResNet, U-Net)
3. Learn transfer learning and fine-tuning
4. Run `03_transfer_learning.py` and `04_image_segmentation_unet.py`
5. Master [TensorFlow Guide](./learn-tensorflow.md) for production deployment
6. Learn distributed training and TensorFlow Serving
7. Explore [NLP and Transformers](./learn-nlp-transformers.md) for text processing
8. Study [Reinforcement Learning](./learn-reinforcement-learning.md) for decision-making
9. Learn [PyTorch](./learn-pytorch.md) as alternative framework
10. Master [Time Series Forecasting](./learn-time-series.md) for temporal data
11. Implement [MLOps](./learn-mlops.md) practices for production
12. Master [Data Science](./learn-data-science.md) role and workflows
13. Deepen [Python](./learn-python-deep.md) programming skills
14. Build [Agentic AI](./learn-agentic-ai.md) systems with LangChain, LlamaIndex, and CrewAI

## 🎯 Key Concepts Covered

### Machine Learning
- Supervised/Unsupervised/Reinforcement Learning
- Classification and Regression
- Feature Engineering
- Model Evaluation
- Hyperparameter Tuning
- Model Deployment

### Deep Learning
- Neural Networks
- Activation Functions
- Backpropagation
- Optimization Algorithms
- Regularization
- Transfer Learning

### CNNs
- Convolution Operations
- Pooling Layers
- CNN Architectures
- Object Detection
- Image Segmentation
- Transfer Learning for Vision

### TensorFlow
- Tensor Operations and Eager Execution
- Model Building APIs (Sequential, Functional, Subclassing)
- Training Pipelines and Custom Training Loops
- Data Loading and Preprocessing
- Custom Components (Layers, Models, Losses, Metrics)
- Distributed Training and Optimization
- Model Deployment and Serving

### NLP & Transformers
- Text Preprocessing and Tokenization
- Word Embeddings and Traditional NLP
- Attention Mechanisms
- Transformer Architecture (BERT, GPT)
- Fine-tuning Pre-trained Models
- Sentiment Analysis, NER, QA

### Reinforcement Learning
- MDP and Value Functions
- Q-Learning and DQN
- Policy Gradients and PPO
- Actor-Critic Methods
- Exploration vs Exploitation

### PyTorch
- Dynamic Computation Graphs
- Autograd and Custom Functions
- nn.Module and Custom Layers
- Data Loading and Training
- Distributed Training
- Model Deployment

### Time Series
- Time Series Components
- Traditional Methods (ARIMA, Prophet)
- LSTM and RNN for Forecasting
- Transformer for Time Series
- Multi-step Forecasting

### MLOps
- CI/CD for ML
- Model Monitoring and Drift Detection
- Model Deployment and Serving
- Experiment Tracking
- Model Registry

### Data Science
- Role and Responsibilities
- Data Science Workflow (CRISP-DM)
- Exploratory Data Analysis
- Statistical Analysis
- Data Visualization
- Communication and Storytelling
- Career Development

### Deep Python
- Advanced Data Structures
- Object-Oriented Programming
- Functional Programming
- Decorators and Metaclasses
- Generators and Iterators
- Concurrency and Parallelism
- Performance Optimization
- Design Patterns

### Agentic AI
- Autonomous Agent Architecture
- LangChain Framework
- LlamaIndex Framework
- CrewAI Multi-Agent Systems
- Tool Integration and Function Calling
- Memory and State Management
- Agent Communication Patterns
- Custom Agent Development

## 🛠️ Tools & Frameworks

- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning library
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Docker**: Containerization
- **MLflow**: Model versioning

## 📝 Best Practices

1. **Start Simple**: Begin with basic models before complex architectures
2. **Data Quality**: Clean and validate data before training
3. **Regularization**: Prevent overfitting with dropout, L2, etc.
4. **Transfer Learning**: Leverage pre-trained models
5. **Monitor Training**: Use callbacks and TensorBoard
6. **Version Control**: Track experiments and models
7. **Evaluate Properly**: Use appropriate metrics and validation strategies

## 🔗 Additional Resources

### Learning Resources
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai](https://www.fast.ai/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Deep Learning Book](https://www.deeplearningbook.org/)

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Documentation
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)

## 📊 Example Use Cases

- **Image Classification**: CIFAR-10, Custom datasets
- **Object Detection**: YOLO, R-CNN
- **Image Segmentation**: Medical imaging, Autonomous vehicles
- **Text Classification**: Sentiment analysis, Spam detection
- **Transfer Learning**: Custom datasets with pre-trained models

## 🤝 Contributing

Feel free to add more examples, improve documentation, or fix issues!

## 📄 License

This documentation is for educational purposes.

---

**Happy Learning! 🎓**

