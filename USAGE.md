# Usage Guide

This guide provides instructions on how to use and run the notebooks in this collection.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 8GB RAM (16GB recommended for deep learning)
- GPU recommended for neural network training

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-ml-mastery-collection.git
cd advanced-ml-mastery-collection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
torch>=1.10.0
transformers>=4.15.0
torchvision>=0.11.0
plotly>=5.4.0
spacy>=3.2.0
prophet>=1.0.0
jupyter>=1.0.0
```

---

## 📓 Running Notebooks

### Jupyter Lab (Recommended)

```bash
jupyter lab
```

Then navigate to the desired notebook in the file browser.

### Jupyter Notebook

```bash
jupyter notebook
```

### Google Colab

1. Open Google Colab: https://colab.research.google.com
2. Click "Upload" and select the notebook file
3. Enable GPU: Runtime → Change runtime type → GPU

---

## 📁 Project Structure

```
advanced-ml-mastery-collection/
├── 01-Regression/
│   ├── Linear-Regression-Real-Estate.ipynb
│   ├── Logistic-Regression-Healthcare.ipynb
│   └── Polynomial-Regression-Energy.ipynb
├── 02-Classification/
│   ├── Random-Forest-Finance.ipynb
│   ├── SVM-Handwritten-Digits.ipynb
│   └── XGBoost-Customer-Churn.ipynb
├── 03-Clustering/
│   ├── K-Means-Segmentation.ipynb
│   ├── DBSCAN-Anomaly-Detection.ipynb
│   └── Hierarchical-Clustering-Genes.ipynb
├── 04-NLP/
│   ├── BERT-Sentiment-Analysis.ipynb
│   ├── GPT-Fine-Tuning-Basics.ipynb
│   └── Named-Entity-Recognition-Spacy.ipynb
├── 05-Computer-Vision/
│   ├── CNN-Medical-Imaging.ipynb
│   ├── YOLOv8-Object-Detection.ipynb
│   ├── ResNet-Transfer-Learning.ipynb
│   ├── GAN-Synthetic-Data-Gen.ipynb
│   ├── Autoencoders-Denoising.ipynb
│   └── Isolation-Forest-Cybersecurity.ipynb
├── 06-Reinforcement-Learning/
│   ├── Q-Learning-Maze-Solver.ipynb
│   ├── Deep-Q-Network-Atari.ipynb
│   ├── RL-CartPole-Agent.ipynb
│   ├── Hyperparameter-Optimization-Optuna.ipynb
│   └── Feature-Engineering-Pipeline.ipynb
├── 07-Time-Series/
│   ├── ARIMA-Sales-Forecasting.ipynb
│   ├── LSTM-Stock-Prediction.ipynb
│   └── Prophet-Market-Trends.ipynb
├── 08-Anomaly-Detection/
│   ├── Autoencoders-Network-Security.ipynb
│   ├── Gradient-Boosting-Insurance.ipynb
│   ├── Isolation-Forest-Cyber.ipynb
│   ├── KNN-Recommender-Systems.ipynb
│   ├── Local-Outlier-Factor-Fraud.ipynb
│   └── Naive-Bayes-Spam-Filter.ipynb
├── 09-Generative-AI/
│   ├── GAN-Face-Generation.ipynb
│   ├── Variational-Autoencoders-MNIST.ipynb
│   ├── Stable-Diffusion-Prompt-Eng.ipynb
│   ├── Transfer-Learning-ResNet.ipynb
│   └── Object-Detection-YOLOv8.ipynb
├── 10-MLOps/
│   ├── Model-Quantization-TensorRT.ipynb
│   ├── Prometheus-ML-Monitoring.ipynb
│   └── Edge-Computing-TinyML.ipynb
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── FEATURES.md
├── USAGE.md
└── CONTRIBUTING.md
```

---

## 💡 Tips for Success

### Data Handling
- Always split data: train (70%), validation (15%), test (15%)
- Normalize/standardize features before training
- Handle missing values appropriately

### Model Training
- Start with simple models, then increase complexity
- Use cross-validation for robust evaluation
- Monitor for overfitting with validation metrics

### Visualization
- Plot learning curves to diagnose issues
- Use confusion matrices for classification
- Visualize feature importance for interpretability

---

## 🐛 Common Issues

### Memory Errors
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Use gradient accumulation
accumulation_steps = 4
```

### Slow Training
- Use GPU acceleration
- Reduce dataset size for prototyping
- Use early stopping

### Import Errors
```bash
# Install missing package
pip install package-name

# Or reinstall all packages
pip install -r requirements.txt --force-reinstall
```
