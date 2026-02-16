# Logistic Regression - Healthcare Diagnosis

**Folder:** `01-Regression-Analysis`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Binary classification for medical diagnosis using logistic regression

## Key Features

- Medical dataset preprocessing
- Feature selection and engineering
- Logistic regression with regularization
- ROC curve and AUC calculation
- Confusion matrix analysis
- Model interpretability

## How to Use

Run the notebook to train a diagnostic model. Adjust hyperparameters for better performance.

## Expected Outputs

Classification metrics, ROC curve, confusion matrix, feature importance

## Difficulty Level

- **Beginner**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 01-Regression-Analysis/Logistic-Regression-Healthcare.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression-Analysis/Logistic-Regression-Healthcare.ipynb --output Logistic-Regression-Healthcare.ipynb --output-dir 01-Regression-Analysis
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
