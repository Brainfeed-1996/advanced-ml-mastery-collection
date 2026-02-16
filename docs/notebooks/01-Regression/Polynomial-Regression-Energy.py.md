# Polynomial Regression - Energy Consumption

**Folder:** `01-Regression`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Predict energy consumption using polynomial regression

## Key Features

- Time series data preprocessing
- Polynomial feature generation
- Overfitting detection and prevention
- Cross-validation for model selection
- Visualization of polynomial fits

## How to Use

Run the notebook to train polynomial models. Compare different polynomial degrees.

## Expected Outputs

Multiple polynomial models, cross-validation scores, prediction plots

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 01-Regression/Polynomial-Regression-Energy.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression/Polynomial-Regression-Energy.ipynb --output Polynomial-Regression-Energy.ipynb --output-dir 01-Regression
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
