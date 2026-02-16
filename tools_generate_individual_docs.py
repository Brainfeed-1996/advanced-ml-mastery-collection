"""Generate individual .md documentation files for each notebook and .py file.

This script creates detailed documentation for:
- Each .ipynb notebook (what it does, how to use it, expected outputs)
- Each .py file (what it does, how to use it, dependencies)

Run:
  python tools_generate_individual_docs.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Notebook descriptions by filename
NOTEBOOK_DESC = {
    # 01-Regression
    "Linear-Regression-Real-Estate": {
        "title": "Linear Regression - Real Estate Price Prediction",
        "purpose": "Predict house prices using linear regression with feature engineering",
        "key_features": [
            "Data preprocessing and cleaning",
            "Feature scaling (StandardScaler, MinMaxScaler)",
            "Linear regression model training",
            "Model evaluation (MSE, R², MAE)",
            "Visualization of predictions vs actual values",
            "Residual analysis"
        ],
        "how_to_use": "Run the notebook to train the model on real estate data. The notebook will output metrics and visualizations.",
        "expected_outputs": "Trained model, evaluation metrics, prediction plots, residual plots",
        "difficulty": "Beginner"
    },
    "Logistic-Regression-Healthcare": {
        "title": "Logistic Regression - Healthcare Diagnosis",
        "purpose": "Binary classification for medical diagnosis using logistic regression",
        "key_features": [
            "Medical dataset preprocessing",
            "Feature selection and engineering",
            "Logistic regression with regularization",
            "ROC curve and AUC calculation",
            "Confusion matrix analysis",
            "Model interpretability"
        ],
        "how_to_use": "Run the notebook to train a diagnostic model. Adjust hyperparameters for better performance.",
        "expected_outputs": "Classification metrics, ROC curve, confusion matrix, feature importance",
        "difficulty": "Beginner"
    },
    "Polynomial-Regression-Energy": {
        "title": "Polynomial Regression - Energy Consumption",
        "purpose": "Predict energy consumption using polynomial regression",
        "key_features": [
            "Time series data preprocessing",
            "Polynomial feature generation",
            "Overfitting detection and prevention",
            "Cross-validation for model selection",
            "Visualization of polynomial fits"
        ],
        "how_to_use": "Run the notebook to train polynomial models. Compare different polynomial degrees.",
        "expected_outputs": "Multiple polynomial models, cross-validation scores, prediction plots",
        "difficulty": "Intermediate"
    },
    
    # 01-Regression-Analysis
    "Random-Forest-Finance": {
        "title": "Random Forest - Financial Risk Assessment",
        "purpose": "Credit risk assessment using Random Forest ensemble method",
        "key_features": [
            "Financial dataset with imbalanced classes",
            "Feature importance analysis",
            "Random Forest hyperparameter tuning",
            "Cross-validation strategies",
            "Handling class imbalance"
        ],
        "how_to_use": "Run the notebook to train a credit risk model. Use feature importance to understand key factors.",
        "expected_outputs": "Trained Random Forest model, feature importance plot, classification report",
        "difficulty": "Intermediate"
    },
    
    # 02-Classification
    "SVM-Handwritten-Digits": {
        "title": "SVM - Handwritten Digit Recognition",
        "purpose": "Multi-class classification of handwritten digits using Support Vector Machines",
        "key_features": [
            "MNIST dataset preprocessing",
            "SVM with different kernels (linear, RBF, polynomial)",
            "Hyperparameter tuning with GridSearchCV",
            "Multi-class classification strategies",
            "Visualization of decision boundaries"
        ],
        "how_to_use": "Run the notebook to train SVM models on digit recognition. Compare kernel performance.",
        "expected_outputs": "Trained SVM models, accuracy scores, confusion matrix, decision boundary plots",
        "difficulty": "Intermediate"
    },
    "XGBoost-Customer-Churn": {
        "title": "XGBoost - Customer Churn Prediction",
        "purpose": "Predict customer churn using gradient boosting (XGBoost)",
        "key_features": [
            "Customer behavior dataset analysis",
            "Feature engineering for churn prediction",
            "XGBoost model training and tuning",
            "Early stopping for prevention of overfitting",
            "Model interpretation with SHAP values"
        ],
        "how_to_use": "Run the notebook to train a churn prediction model. Use SHAP for model interpretation.",
        "expected_outputs": "Trained XGBoost model, feature importance, SHAP plots, churn predictions",
        "difficulty": "Advanced"
    },
    
    # 02-Classification-Challenges
    "K-Means-Customer-Segmentation": {
        "title": "K-Means - Customer Segmentation",
        "purpose": "Segment customers into groups using K-Means clustering",
        "key_features": [
            "Customer behavior clustering",
            "Elbow method for K selection",
            "Silhouette analysis",
            "Cluster profiling and interpretation",
            "Visualization of customer segments"
        ],
        "how_to_use": "Run the notebook to segment customers. Adjust K based on business needs.",
        "expected_outputs": "Customer segments, cluster profiles, visualization plots",
        "difficulty": "Intermediate"
    },
    "SVM-Image-Recognition": {
        "title": "SVM - Image Recognition",
        "purpose": "Image classification using Support Vector Machines",
        "key_features": [
            "Image feature extraction",
            "SVM for image classification",
            "Data augmentation techniques",
            "Model evaluation on test set",
            "Visualization of predictions"
        ],
        "how_to_use": "Run the notebook to train an image classifier. Use data augmentation for better performance.",
        "expected_outputs": "Trained SVM model, accuracy metrics, prediction examples",
        "difficulty": "Intermediate"
    },
    
    # 03-Clustering
    "DBSCAN-Anomaly-Detection": {
        "title": "DBSCAN - Anomaly Detection",
        "purpose": "Detect anomalies using density-based clustering (DBSCAN)",
        "key_features": [
            "Anomaly detection in high-dimensional data",
            "DBSCAN parameter tuning",
            "Outlier detection and visualization",
            "Comparison with other clustering methods",
            "Real-world anomaly detection patterns"
        ],
        "how_to_use": "Run the notebook to detect anomalies. Adjust epsilon and min_samples parameters.",
        "expected_outputs": "Anomaly scores, outlier detection, visualization of normal vs anomalous data",
        "difficulty": "Advanced"
    },
    "Hierarchical-Clustering-Genes": {
        "title": "Hierarchical Clustering - Gene Expression",
        "purpose": "Cluster genes using hierarchical clustering for biological analysis",
        "key_features": [
            "Gene expression data preprocessing",
            "Hierarchical clustering with different linkage methods",
            "Dendrogram visualization and interpretation",
            "Cluster validation metrics",
            "Biological pathway analysis"
        ],
        "how_to_use": "Run the notebook to cluster genes. Interpret clusters in biological context.",
        "expected_outputs": "Gene clusters, dendrogram, cluster validation metrics",
        "difficulty": "Advanced"
    },
    "K-Means-Segmentation": {
        "title": "K-Means - Image Segmentation",
        "purpose": "Segment images using K-Means clustering",
        "key_features": [
            "Image color space conversion",
            "K-Means for color segmentation",
            "Segmentation quality evaluation",
            "Comparison with other segmentation methods",
            "Application to real images"
        ],
        "how_to_use": "Run the notebook to segment images. Adjust K for different segmentation levels.",
        "expected_outputs": "Segmented images, segmentation metrics, comparison plots",
        "difficulty": "Intermediate"
    },
    
    # 03-Clustering-Techniques
    "BERT-Sentiment-Analysis": {
        "title": "BERT - Sentiment Analysis",
        "purpose": "Sentiment classification using BERT transformer model",
        "key_features": [
            "Text preprocessing and tokenization",
            "BERT model fine-tuning",
            "Sentiment classification pipeline",
            "Model evaluation on test data",
            "Inference on new text"
        ],
        "how_to_use": "Run the notebook to train a sentiment classifier. Use the trained model for inference.",
        "expected_outputs": "Trained BERT model, accuracy metrics, sentiment predictions",
        "difficulty": "Advanced"
    },
    "LSTM-Stock-Prediction": {
        "title": "LSTM - Stock Price Prediction",
        "purpose": "Predict stock prices using LSTM neural networks",
        "key_features": [
            "Time series data preprocessing",
            "LSTM architecture design",
            "Sequence modeling for prediction",
            "Model training with callbacks",
            "Forecasting future prices"
        ],
        "how_to_use": "Run the notebook to train an LSTM model. Use it for stock price forecasting.",
        "expected_outputs": "Trained LSTM model, prediction plots, forecast metrics",
        "difficulty": "Advanced"
    },
    "PCA-Dimensionality-Reduction": {
        "title": "PCA - Dimensionality Reduction",
        "purpose": "Reduce dimensionality using Principal Component Analysis",
        "key_features": [
            "High-dimensional data analysis",
            "PCA implementation and interpretation",
            "Variance explained by components",
            "Dimensionality reduction for visualization",
            "Application to downstream tasks"
        ],
        "how_to_use": "Run the notebook to reduce dimensionality. Use reduced features for other ML tasks.",
        "expected_outputs": "Reduced dimensionality data, explained variance plots, component analysis",
        "difficulty": "Intermediate"
    },
    
    # 04-NLP
    "CNN-Medical-Imaging": {
        "title": "CNN - Medical Imaging Diagnosis",
        "purpose": "Medical image classification using Convolutional Neural Networks",
        "key_features": [
            "Medical image preprocessing",
            "CNN architecture for medical imaging",
            "Transfer learning with pre-trained models",
            "Model evaluation on medical data",
            "Interpretability techniques"
        ],
        "how_to_use": "Run the notebook to train a medical image classifier. Use transfer learning for better performance.",
        "expected_outputs": "Trained CNN model, accuracy metrics, prediction examples",
        "difficulty": "Advanced"
    },
    "GPT-Fine-Tuning-Basics": {
        "title": "GPT - Fine-Tuning Basics",
        "purpose": "Fine-tune GPT model for text generation tasks",
        "key_features": [
            "Text dataset preparation",
            "GPT model fine-tuning",
            "Text generation pipeline",
            "Model evaluation metrics",
            "Inference examples"
        ],
        "how_to_use": "Run the notebook to fine-tune GPT. Use the fine-tuned model for text generation.",
        "expected_outputs": "Fine-tuned GPT model, generated text examples, evaluation metrics",
        "difficulty": "Advanced"
    },
    
    # 04-Natural-Language-Processing
    "Named-Entity-Recognition": {
        "title": "Named Entity Recognition - spaCy",
        "purpose": "Extract named entities from text using spaCy",
        "key_features": [
            "Text preprocessing with spaCy",
            "Named entity recognition pipeline",
            "Custom entity recognition",
            "Entity visualization",
            "Application to real documents"
        ],
        "how_to_use": "Run the notebook to extract entities. Customize for specific entity types.",
        "expected_outputs": "Extracted entities, entity visualization, custom entity models",
        "difficulty": "Intermediate"
    },
    
    # 05-Computer-Vision
    "ResNet-Transfer-Learning": {
        "title": "ResNet - Transfer Learning",
        "purpose": "Transfer learning with ResNet for image classification",
        "key_features": [
            "Pre-trained ResNet model loading",
            "Transfer learning implementation",
            "Fine-tuning strategies",
            "Feature extraction",
            "Model evaluation"
        ],
        "how_to_use": "Run the notebook to apply transfer learning. Use pre-trained weights for faster training.",
        "expected_outputs": "Fine-tuned ResNet model, accuracy metrics, prediction examples",
        "difficulty": "Intermediate"
    },
    "YOLOv8-Object-Detection": {
        "title": "YOLOv8 - Object Detection",
        "purpose": "Real-time object detection using YOLOv8",
        "key_features": [
            "YOLOv8 model loading and inference",
            "Object detection pipeline",
            "Real-time detection setup",
            "Model evaluation metrics",
            "Application to video streams"
        ],
        "how_to_use": "Run the notebook to detect objects. Use webcam for real-time detection.",
        "expected_outputs": "Detected objects, bounding boxes, confidence scores, real-time detection",
        "difficulty": "Advanced"
    },
    
    # 06-Reinforcement-Learning
    "RL-CartPole-Agent": {
        "title": "RL - CartPole Agent",
        "purpose": "Train a reinforcement learning agent to solve CartPole environment",
        "key_features": [
            "Gymnasium environment setup",
            "RL agent training",
            "Policy gradient methods",
            "Training visualization",
            "Agent evaluation"
        ],
        "how_to_use": "Run the notebook to train an RL agent. Monitor training progress.",
        "expected_outputs": "Trained RL agent, training curves, agent performance metrics",
        "difficulty": "Advanced"
    },
    "Deep-Q-Network-Atari": {
        "title": "DQN - Atari Game Playing",
        "purpose": "Train a Deep Q-Network to play Atari games",
        "key_features": [
            "Atari game environment setup",
            "DQN architecture implementation",
            "Experience replay",
            "Target network",
            "Training and evaluation"
        ],
        "how_to_use": "Run the notebook to train a DQN agent. This is resource-intensive.",
        "expected_outputs": "Trained DQN agent, game play videos, performance metrics",
        "difficulty": "Advanced"
    },
    "Q-Learning-Maze-Solver": {
        "title": "Q-Learning - Maze Solver",
        "purpose": "Solve mazes using Q-Learning algorithm",
        "key_features": [
            "Maze environment creation",
            "Q-Learning implementation",
            "Exploration vs exploitation",
            "Policy visualization",
            "Solution evaluation"
        ],
        "how_to_use": "Run the notebook to train a maze solver. Adjust learning parameters.",
        "expected_outputs": "Trained Q-table, solution path, convergence plots",
        "difficulty": "Intermediate"
    },
    "Hyperparameter-Optimization-Optuna": {
        "title": "Hyperparameter Optimization - Optuna",
        "purpose": "Optimize hyperparameters using Optuna framework",
        "key_features": [
            "Optuna study setup",
            "Hyperparameter search space definition",
            "Pruning strategies",
            "Best parameters extraction",
            "Visualization of optimization"
        ],
        "how_to_use": "Run the notebook to optimize hyperparameters. Use results for model training.",
        "expected_outputs": "Best hyperparameters, optimization history, parameter importance",
        "difficulty": "Intermediate"
    },
    "Prophet-Market-Trends": {
        "title": "Prophet - Market Trends Forecasting",
        "purpose": "Forecast market trends using Facebook Prophet",
        "key_features": [
            "Time series data preparation",
            "Prophet model configuration",
            "Trend and seasonality analysis",
            "Forecast generation",
            "Model evaluation"
        ],
        "how_to_use": "Run the notebook to generate forecasts. Adjust seasonality parameters.",
        "expected_outputs": "Trend forecasts, seasonality plots, prediction intervals",
        "difficulty": "Intermediate"
    },
    
    # 07-Time-Series
    "Decision-Trees-Interpretability": {
        "title": "Decision Trees - Model Interpretability",
        "purpose": "Interpret decision tree models using visualization techniques",
        "key_features": [
            "Decision tree visualization",
            "Feature importance analysis",
            "Tree interpretation techniques",
            "Model explanation methods",
            "Comparison with other models"
        ],
        "how_to_use": "Run the notebook to interpret decision trees. Use insights for model improvement.",
        "expected_outputs": "Tree visualizations, feature importance, model explanations",
        "difficulty": "Intermediate"
    },
    "ML-Model-Monitoring-Prometheus": {
        "title": "ML Model Monitoring - Prometheus",
        "purpose": "Monitor ML model performance using Prometheus metrics",
        "key_features": [
            "Model performance tracking",
            "Prometheus metrics setup",
            "Drift detection",
            "Alert configuration",
            "Dashboard creation"
        ],
        "how_to_use": "Run the notebook to set up model monitoring. Integrate with production systems.",
        "expected_outputs": "Monitoring metrics, drift alerts, performance dashboards",
        "difficulty": "Advanced"
    },
    "Model-Quantization-TensorRT": {
        "title": "Model Quantization - TensorRT",
        "purpose": "Quantize ML models for deployment using TensorRT",
        "key_features": [
            "Model quantization techniques",
            "TensorRT optimization",
            "Inference speed improvement",
            "Model accuracy preservation",
            "Deployment preparation"
        ],
        "how_to_use": "Run the notebook to quantize models. Use quantized models for deployment.",
        "expected_outputs": "Quantized models, speed benchmarks, accuracy comparison",
        "difficulty": "Advanced"
    },
    
    # 07-Time-Series-Forecasting
    "Feature-Engineering-Pipeline": {
        "title": "Feature Engineering Pipeline",
        "purpose": "Build automated feature engineering pipelines for ML",
        "key_features": [
            "Feature engineering automation",
            "Pipeline construction",
            "Feature selection techniques",
            "Pipeline evaluation",
            "Production-ready pipelines"
        ],
        "how_to_use": "Run the notebook to build feature engineering pipelines. Use in production workflows.",
        "expected_outputs": "Feature engineering pipeline, engineered features, pipeline metrics",
        "difficulty": "Advanced"
    },
    
    # 08-Anomaly-Detection
    "Variational-Autoencoders-MNIST": {
        "title": "Variational Autoencoders - MNIST",
        "purpose": "Generate and detect anomalies using VAE on MNIST",
        "key_features": [
            "VAE architecture implementation",
            "Anomaly detection with reconstruction error",
            "Latent space visualization",
            "Image generation from latent space",
            "Anomaly scoring"
        ],
        "how_to_use": "Run the notebook to detect anomalies. Use reconstruction error for scoring.",
        "expected_outputs": "Trained VAE, anomaly scores, generated images, latent space plots",
        "difficulty": "Advanced"
    },
    
    # 09-Generative-AI
    "Stable-Diffusion-Prompt-Eng": {
        "title": "Stable Diffusion - Prompt Engineering",
        "purpose": "Generate images using Stable Diffusion with prompt engineering",
        "key_features": [
            "Stable Diffusion model loading",
            "Prompt engineering techniques",
            "Image generation pipeline",
            "Parameter tuning",
            "Quality evaluation"
        ],
        "how_to_use": "Run the notebook to generate images. Use CPU-safe default or full model.",
        "expected_outputs": "Generated images, prompt variations, quality metrics",
        "difficulty": "Intermediate"
    },
    
    # 10-MLOps
    "Federated-Learning-Privacy": {
        "title": "Federated Learning - Privacy Preservation",
        "purpose": "Implement federated learning with privacy guarantees",
        "key_features": [
            "Federated learning setup",
            "Privacy-preserving techniques",
            "Distributed model training",
            "Privacy metrics",
            "Security considerations"
        ],
        "how_to_use": "Run the notebook to implement federated learning. Use for privacy-sensitive applications.",
        "expected_outputs": "Federated model, privacy metrics, distributed training logs",
        "difficulty": "Advanced"
    },
    "ML-Edge-Computing-TinyML": {
        "title": "ML Edge Computing - TinyML",
        "purpose": "Deploy ML models on edge devices using TinyML techniques",
        "key_features": [
            "Model compression techniques",
            "Edge device deployment",
            "TinyML optimization",
            "Performance evaluation",
            "Real-world applications"
        ],
        "how_to_use": "Run the notebook to prepare models for edge deployment. Use optimized models on devices.",
        "expected_outputs": "Optimized models, deployment scripts, performance metrics",
        "difficulty": "Advanced"
    },
    "Transformer-Attention-Mechanisms": {
        "title": "Transformer - Attention Mechanisms",
        "purpose": "Understand and implement transformer attention mechanisms",
        "key_features": [
            "Attention mechanism visualization",
            "Transformer architecture",
            "Self-attention implementation",
            "Multi-head attention",
            "Application to NLP tasks"
        ],
        "how_to_use": "Run the notebook to understand attention. Use insights for transformer development.",
        "expected_outputs": "Attention visualizations, transformer components, implementation examples",
        "difficulty": "Advanced"
    },
    "Edge-Computing-TinyML": {
        "title": "Edge Computing - TinyML Production",
        "purpose": "Production-ready TinyML deployment for edge devices",
        "key_features": [
            "Production deployment pipeline",
            "Model optimization for edge",
            "Performance monitoring",
            "Update mechanisms",
            "Real-world deployment patterns"
        ],
        "how_to_use": "Run the notebook for production edge deployment. Follow deployment patterns.",
        "expected_outputs": "Production-ready models, deployment scripts, monitoring setup",
        "difficulty": "Advanced"
    },
    "Prometheus-ML-Monitoring": {
        "title": "Prometheus - ML Monitoring",
        "purpose": "Comprehensive ML model monitoring with Prometheus",
        "key_features": [
            "Model performance metrics",
            "Drift detection algorithms",
            "Alert configuration",
            "Dashboard creation",
            "Integration with monitoring systems"
        ],
        "how_to_use": "Run the notebook to set up ML monitoring. Integrate with production systems.",
        "expected_outputs": "Monitoring metrics, alerts, dashboards, drift detection",
        "difficulty": "Advanced"
    },
}

# Tool file descriptions
TOOL_DESC = {
    "tools_execute_batch": "Batch execution of multiple notebooks with timeout and error handling.",
    "tools_fix_concatenated_imports": "Fix notebooks with concatenated imports (single-line imports).",
    "tools_fix_ipynb_json": "Repair corrupted JSON structure in notebook files.",
    "tools_rebuild": "Rebuild notebooks with industrial-grade patterns and outputs.",
    "tools_generate_readmes": "Auto-generate comprehensive README files for all modules.",
    "tools_repair": "Repair specific notebooks with targeted fixes.",
    "tools_rewrite": "Rewrite specific cells in notebooks.",
    "tools_patch": "Patch notebook cells for compatibility.",
    "tools_force_outputs": "Force outputs in notebooks with allow_errors.",
    "tools_notebooks_report": "Generate report on notebook status.",
    "check_nb": "Check notebook health and structure.",
    "validate_notebooks": "Validate all notebooks for outputs and structure.",
}

def generate_notebook_doc(nb_name: str, folder_name: str) -> str:
    """Generate documentation for a specific notebook."""
    
    # Find the notebook description
    desc = NOTEBOOK_DESC.get(nb_name, {
        "title": nb_name,
        "purpose": "Notebook project (see notebook for details)",
        "key_features": ["See notebook for details"],
        "how_to_use": "Run the notebook to execute the project",
        "expected_outputs": "See notebook for outputs",
        "difficulty": "Unknown"
    })
    
    lines = []
    lines += [f"# {desc['title']}", ""]
    lines += [f"**Folder:** `{folder_name}`", ""]
    lines += [f"**Last updated:** {STAMP}", ""]
    lines += ["## Purpose", "", desc['purpose'], ""]
    
    lines += ["## Key Features", ""]
    for feature in desc['key_features']:
        lines += [f"- {feature}"]
    lines += [""]
    
    lines += ["## How to Use", "", desc['how_to_use'], ""]
    
    lines += ["## Expected Outputs", "", desc['expected_outputs'], ""]
    
    lines += ["## Difficulty Level", "", f"- **{desc['difficulty']}**", ""]
    
    lines += ["## Prerequisites", "", "- Python 3.10+", "- Virtual environment (see docs/INSTALLATION.md)", "- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)", ""]
    
    lines += ["## Running the Notebook", "", "### Interactive", "```bash", f"jupyter notebook {folder_name}/{nb_name}.ipynb", "```", "", "### Headless (with outputs)", "```bash", "python -m jupyter nbconvert --to notebook --execute \\", f"  {folder_name}/{nb_name}.ipynb --output {nb_name}.ipynb --output-dir {folder_name}", "```", ""]
    
    lines += ["## Troubleshooting", "", "- **Missing outputs**: Run the notebook in Jupyter or use nbconvert", "- **Memory issues**: Reduce batch size or use smaller models", "- **Slow execution**: Use CPU-safe defaults or reduce complexity", "- **Dependency errors**: Install required packages from requirements.txt", ""]
    
    return "\n".join(lines).replace("\r\n", "\n")


def generate_tool_doc(py_name: str, folder_name: str) -> str:
    """Generate documentation for a specific Python tool."""
    
    # Find the tool description
    desc = TOOL_DESC.get(py_name, "Automation tool for notebook management.")
    
    lines = []
    lines += [f"# {py_name}.py", ""]
    lines += [f"**Folder:** `{folder_name}`", ""]
    lines += [f"**Last updated:** {STAMP}", ""]
    lines += ["## Purpose", "", desc, ""]
    
    lines += ["## How to Use", "", "```bash", f"cd {folder_name}", f"python {py_name}.py", "```", ""]
    
    lines += ["## What It Does", "", "- Processes notebooks in the current folder", "- Applies industrial-grade patterns", "- Fixes common issues", "- Generates outputs", ""]
    
    lines += ["## Dependencies", "", "- Jupyter nbconvert", "- Python 3.10+", "- Standard ML libraries", ""]
    
    lines += ["## Output", "", "- Modified notebooks with outputs", "- Log files (if any)", "- Validation reports", ""]
    
    lines += ["## Notes", "", "- Always backup notebooks before running", "- Check logs for any issues", "- Validate outputs after execution", ""]
    
    return "\n".join(lines).replace("\r\n", "\n")


def main() -> None:
    """Generate documentation for all notebooks and tools."""
    
    # Create docs directory if it doesn't exist
    docs_dir = ROOT / "docs" / "notebooks"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all folders
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("."):
            continue
        if folder.name in {"core", "docs", "github", "__pycache__"}:
            continue
        
        # Create folder-specific docs directory
        folder_docs_dir = docs_dir / folder.name
        folder_docs_dir.mkdir(exist_ok=True)
        
        # Process notebooks (.ipynb files)
        for nb in sorted(folder.glob("*.ipynb")):
            nb_name = nb.stem
            doc_content = generate_notebook_doc(nb_name, folder.name)
            doc_path = folder_docs_dir / f"{nb_name}.md"
            doc_path.write_text(doc_content, encoding="utf-8")
            print(f"Created: {doc_path.relative_to(ROOT)}")
        
        # Process Python scripts (.py files)
        for py in sorted(folder.glob("*.py")):
            py_name = py.stem
            # Skip the documentation script itself
            if py_name == "tools_generate_individual_docs":
                continue
            # Skip automation tools (they have their own documentation)
            if py_name.startswith("tools_") or py_name == "check_nb" or py_name == "validate_notebooks":
                continue
            # Only process .py files that are companion scripts to notebooks
            # Check if there's a corresponding .ipynb file
            if (folder / f"{py_name}.ipynb").exists():
                doc_content = generate_notebook_doc(py_name, folder.name)
                doc_path = folder_docs_dir / f"{py_name}.py.md"
                doc_path.write_text(doc_content, encoding="utf-8")
                print(f"Created: {doc_path.relative_to(ROOT)}")
    
    # Create a master index
    index_lines = ["# Individual Notebook and Tool Documentation", "", f"**Generated:** {STAMP}", "", "## Overview", "", "This directory contains individual documentation files for each notebook and Python tool in the repository.", "", "## Structure", "", "- `docs/notebooks/` - Individual documentation for each notebook", "- `docs/notebooks/<folder>/` - Documentation for notebooks in each folder", "- `docs/notebooks/<folder>/<notebook>.md` - Documentation for specific notebook", "- `docs/notebooks/<folder>/<tool>.md` - Documentation for specific tool", "", "## How to Use", "", "1. Navigate to the folder containing the notebook/tool you want to understand", "2. Read the corresponding .md file", "3. Follow the instructions to run the notebook/tool", "4. Review the expected outputs", "", "## Documentation Quality", "", "Each .md file includes:", "- Purpose and use case", "  - Key features", "  - How to use", "  - Expected outputs", "  - Troubleshooting tips", "  - Prerequisites", "", "## Index", "", "See individual .md files for detailed documentation.", "", "---", "", "Generated by `tools_generate_individual_docs.py`", ""]
    
    index_path = docs_dir / "INDEX.md"
    index_path.write_text("\n".join(index_lines).replace("\r\n", "\n"), encoding="utf-8")
    print(f"Created: {index_path.relative_to(ROOT)}")
    
    print("\n" + "="*60)
    print("Documentation generation complete!")
    print("="*60)
    print(f"Total notebooks documented: {len(NOTEBOOK_DESC)}")
    print(f"Total tools documented: {len(TOOL_DESC)}")
    print(f"Documentation location: {docs_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
