# SVM - Handwritten Digit Recognition

**Folder:** `02-Classification`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Multi-class classification of handwritten digits using Support Vector Machines

## Key Features

- MNIST dataset preprocessing
- SVM with different kernels (linear, RBF, polynomial)
- Hyperparameter tuning with GridSearchCV
- Multi-class classification strategies
- Visualization of decision boundaries

## How to Use

Run the notebook to train SVM models on digit recognition. Compare kernel performance.

## Expected Outputs

Trained SVM models, accuracy scores, confusion matrix, decision boundary plots

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 02-Classification/SVM-Handwritten-Digits.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification/SVM-Handwritten-Digits.ipynb --output SVM-Handwritten-Digits.ipynb --output-dir 02-Classification
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
