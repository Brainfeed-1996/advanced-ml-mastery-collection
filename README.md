# Advanced ML Mastery Collection v2.0

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, industrial-grade collection of 67+ Machine Learning and Deep Learning projects, from Regression to Transformer architectures.

## 🎯 Purpose & Learning Objectives

This repository is designed for **production-ready ML engineering**. Each notebook and tool demonstrates:

- **Industrial data pipelines**: synthetic data generation, preprocessing, feature engineering
- **Model selection & evaluation**: cross-validation, hyperparameter tuning, metrics
- **Visualization & interpretation**: plots, confusion matrices, ROC curves, SHAP
- **Reproducibility**: fixed seeds, deterministic operations, version pinning
- **Scalability**: CPU/GPU considerations, memory optimization, streaming patterns

## 📚 Learning Path

The full learning path is maintained in **docs/LEARNING_PATH.md**.

## 🚀 Quick Start / Installation

See **docs/INSTALLATION.md** for:
- Virtual environment setup
- Core dependencies
- Heavy dependencies (Transformers, Diffusers, Ultralytics, etc.)

## 📊 Project Structure

```
advanced-ml-mastery-collection/
├── 01-Regression/              # Linear, Logistic, Polynomial regression
├── 01-Regression-Analysis/     # Advanced regression diagnostics
├── 02-Classification/          # SVM, Random Forest, XGBoost
├── 02-Classification-Challenges/ # Harder variants, tuning
├── 03-Clustering/              # K-Means, DBSCAN, Hierarchical
├── 03-Clustering-Techniques/   # Advanced clustering + dimensionality reduction
├── 04-NLP/                     # BERT, GPT, NER, TF-IDF
├── 04-Natural-Language-Processing/ # Advanced NLP + RL intro
├── 05-Computer-Vision/         # CNN, Autoencoders, GAN, YOLO, ResNet TL
├── 06-Reinforcement-Learning/  # DQN, Q-Learning, Optuna, Prophet
├── 07-Time-Series/             # LSTM, Prophet, ARIMA
├── 07-Time-Series-Forecasting/ # Monitoring, Quantization, Interpretability
├── 08-Anomaly-Detection/       # Isolation Forest, Streaming drift
├── 09-Generative-AI/           # GAN, VAE, Stable Diffusion, YOLO
├── 10-MLOps/                   # Monitoring, Quantization, Edge ML
├── 10-MLOps-Production/        # Federated, Transformer attention, Edge
├── core/                       # Core baseline notebooks
├── docs/                       # Documentation
├── tools_*.py                  # Automation & industrial tooling
└── requirements.txt
```

## 🛠️ Tools & Automation

This repository includes industrial automation tools (see **docs/TOOLS.md**):

- `tools_execute_batch*.py` - Batch execution of notebooks
- `tools_fix_*.py` - Fix concatenated imports, JSON corruption
- `tools_rebuild_*.py` - Rebuild notebooks with industrial patterns
- `tools_generate_readmes.py` - Auto-generate module READMEs
- `tools_rebuild_stable_diffusion_industrial.py` - CPU-safe SD execution

## 📖 Documentation

- **docs/LEARNING_PATH.md** - Complete learning path
- **docs/INSTALLATION.md** - Setup instructions
- **docs/OPTIONAL_HEAVY_DEPS.md** - Heavy dependencies guide
- **docs/TOOLS.md** - Tooling documentation
- **docs/EXECUTION_INDEX.md** - Execution index

## 🎯 Featured Notebooks

### ⭐ Linear Regression - Real Estate
Predict house prices using linear regression with feature engineering.

**Key features:**
- Data preprocessing
- Feature scaling
- Regularization (Ridge, Lasso)
- Model evaluation (MSE, R²)

### ⭐ Random Forest - Finance
Credit scoring with ensemble methods.

**Key features:**
- Feature importance
- Cross-validation
- Hyperparameter tuning
- Class imbalance handling

### ⭐ LSTM - Stock Prediction
Time series forecasting with deep learning.

**Key features:**
- Sequence preprocessing
- LSTM architecture
- Sliding window approach
- Volatility modeling

### ⭐ BERT - Sentiment Analysis
Transformer-based NLP classification.

**Key features:**
- Tokenization
- Fine-tuning BERT
- Attention visualization
- Model deployment

## 🛠️ Technologies

| Category | Libraries |
|----------|-----------|
| **Frameworks** | TensorFlow, PyTorch, scikit-learn |
| **NLP** | spaCy, Transformers, NLTK |
| **Computer Vision** | OpenCV, YOLO, ResNet |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Time Series** | Prophet, Statsmodels |
| **MLOps** | Optuna, MLflow, Weights & Biases |

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

## 📦 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Brainfeed-1996/advanced-ml-mastery-collection.git
cd advanced-ml-mastery-collection
```

### 2. Create a virtual environment
```bash
python -m venv ml-env
# Windows:
.\ml-env\Scripts\Activate.ps1
# Linux/Mac:
source ml-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a notebook
```bash
jupyter notebook 01-Regression/Linear-Regression-Real-Estate.ipynb
```

### 5. Execute notebooks headlessly (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression/Linear-Regression-Real-Estate.ipynb \
  --output Linear-Regression-Real-Estate.ipynb \
  --output-dir 01-Regression
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your notebook with documentation
4. Ensure all notebooks have outputs
5. Submit a pull request

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 👤 Author

**Olivier Robert-Duboille**  
GitHub: https://github.com/Brainfeed-1996

---

<div align="center">

Made with ❤️ for the ML Community

</div>
