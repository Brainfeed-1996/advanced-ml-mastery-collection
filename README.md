# Advanced ML Mastery Collection v2.0

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of 50+ Machine Learning and Deep Learning projects, from Regression to Transformer architectures.

## 📚 Learning Path

```
Beginner
├── 01-Regression
│   ├── Linear-Regression-Real-Estate.ipynb ⭐
│   ├── Logistic-Regression-Healthcare.ipynb ⭐
│   └── Polynomial-Regression-Energy.ipynb
├── 02-Classification
│   ├── Random-Forest-Finance.ipynb ⭐
│   ├── SVM-Handwritten-Digits.ipynb ⭐
│   └── XGBoost-Customer-Churn.ipynb ⭐
└── 03-Clustering
    ├── K-Means-Customer-Segmentation.ipynb ⭐
    ├── DBSCAN-Anomaly-Detection.ipynb
    └── Hierarchical-Clustering-Genes.ipynb

Intermediate
├── 04-Deep-Learning
│   ├── CNN-Medical-Imaging.ipynb ⭐
│   ├── LSTM-Stock-Prediction.ipynb ⭐
│   └── Autoencoders-Denoising.ipynb
├── 05-NLP
│   ├── BERT-Sentiment-Analysis.ipynb ⭐
│   ├── GPT-Fine-Tuning-Basics.ipynb ⭐
│   └── Named-Entity-Recognition-Spacy.ipynb
└── 06-Time-Series
    ├── ARIMA-Sales-Forecasting.ipynb ⭐
    ├── Prophet-Market-Trends.ipynb
    └── LSTM-Stock-Prediction.ipynb

Advanced
├── 07-Reinforcement-Learning
│   ├── RL-CartPole-Agent.ipynb ⭐
│   ├── Deep-Q-Network-Atari.ipynb
    └── Q-Learning-Maze-Solver.ipynb
├── 08-Generative-AI
│   ├── GAN-Synthetic-Data-Gen.ipynb ⭐
│   ├── GAN-Face-Generation.ipynb
│   └── Variational-Autoencoders-MNIST.ipynb
└── 09-MLOps
    ├── Model-Quantization-TensorRT.ipynb ⭐
    ├── ML-Model-Monitoring-Prometheus.ipynb
    └── Federated-Learning-Privacy.ipynb
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Brainfeed-1996/advanced-ml-mastery-collection.git
cd advanced-ml-mastery-collection

# Install dependencies
pip install -r requirements.txt

# Run a notebook
jupyter notebook 01-Regression/Linear-Regression-Real-Estate.ipynb
```

## 📦 Installation

### Create Virtual Environment

```bash
# Create venv
python -m venv ml-env
source ml-env/bin/activate  # Linux/Mac
# or
ml-env\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### Requirements

```
# Core ML
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0

# Deep Learning (choose one or both)
tensorflow>=2.10
torch>=2.0

# NLP
transformers>=4.30
spacy>=3.5
nltk>=3.8

# Computer Vision
opencv-python>=4.7
torchvision>=0.15

# Visualization
matplotlib>=3.5
seaborn>=0.11
plotly>=5.10

# Time Series
prophet>=1.1
statsmodels>=0.13

# MLOps
optuna>=3.0
mlflow>=2.0
```

## 📊 Project Structure

```
advanced-ml-mastery-collection/
├── 01-Regression/              # Linear, Logistic, Polynomial
├── 02-Classification/           # SVM, Random Forest, XGBoost
├── 03-Clustering/              # K-Means, DBSCAN, Hierarchical
├── 04-Deep-Learning/           # CNN, LSTM, Autoencoders
├── 05-NLP/                     # BERT, GPT, NER
├── 06-Time-Series/             # ARIMA, Prophet, LSTM
├── 07-Reinforcement-Learning/  # DQN, PPO, Q-Learning
├── 08-Generative-AI/           # GANs, VAEs, Stable Diffusion
├── 09-MLOps/                   # Quantization, Monitoring
├── requirements.txt
└── README.md
```

## 🎯 Featured Notebooks

### ⭐ Linear Regression - Real Estate

Predict house prices using linear regression with feature engineering.

```python
# Key features
- Data preprocessing
- Feature scaling
- Regularization (Ridge, Lasso)
- Model evaluation (MSE, R²)
```

### ⭐ Random Forest - Finance

Credit scoring with ensemble methods.

```python
# Key features
- Feature importance
- Cross-validation
- Hyperparameter tuning
- Class imbalance handling
```

### ⭐ LSTM - Stock Prediction

Time series forecasting with deep learning.

```python
# Key features
- Sequence preprocessing
- LSTM architecture
- Sliding window approach
- Volatility modeling
```

### ⭐ BERT - Sentiment Analysis

Transformer-based NLP classification.

```python
# Key features
- Tokenization
- Fine-tuning BERT
- Attention visualization
- Model deployment
```

## 🛠️ Technologies

| Category | Libraries |
|----------|-----------|
| **Frameworks** | TensorFlow, PyTorch, scikit-learn |
| **NLP** | spaCy, Transformers, NLTK |
| **Computer Vision** | OpenCV, YOLO, ResNet |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Time Series** | Prophet, Statsmodels |
| **MLOps** | Optuna, MLflow, Weights & Biases |

## 📖 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[FEATURES.md](docs/FEATURES.md)** - Complete feature catalog
- **[USAGE.md](docs/USAGE.md)** - Usage guides

## 🏆 Learning Outcomes

After completing this collection, you will master:

- [ ] Data preprocessing and feature engineering
- [ ] Supervised learning algorithms
- [ ] Unsupervised learning techniques
- [ ] Deep neural networks (CNN, RNN, Transformers)
- [ ] Natural Language Processing
- [ ] Time series analysis
- [ ] Reinforcement learning
- [ ] Model deployment and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your notebook with documentation
4. Submit a pull request

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 👤 Author

**Olivier Robert-Duboille**  
GitHub: https://github.com/Brainfeed-1996

---

<div align="center">

Made with ❤️ for the ML Community

</div>
