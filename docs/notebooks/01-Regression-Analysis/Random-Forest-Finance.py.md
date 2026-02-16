# Random Forest - Financial Risk Assessment

**Folder:** `01-Regression-Analysis`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Credit risk assessment using Random Forest ensemble method

## Key Features

- Financial dataset with imbalanced classes
- Feature importance analysis
- Random Forest hyperparameter tuning
- Cross-validation strategies
- Handling class imbalance

## How to Use

Run the notebook to train a credit risk model. Use feature importance to understand key factors.

## Expected Outputs

Trained Random Forest model, feature importance plot, classification report

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 01-Regression-Analysis/Random-Forest-Finance.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression-Analysis/Random-Forest-Finance.ipynb --output Random-Forest-Finance.ipynb --output-dir 01-Regression-Analysis
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
