# Linear Regression - Real Estate Price Prediction

**Folder:** `01-Regression-Analysis`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Predict house prices using linear regression with feature engineering

## Key Features

- Data preprocessing and cleaning
- Feature scaling (StandardScaler, MinMaxScaler)
- Linear regression model training
- Model evaluation (MSE, R², MAE)
- Visualization of predictions vs actual values
- Residual analysis

## How to Use

Run the notebook to train the model on real estate data. The notebook will output metrics and visualizations.

## Expected Outputs

Trained model, evaluation metrics, prediction plots, residual plots

## Difficulty Level

- **Beginner**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 01-Regression-Analysis/Linear-Regression-Real-Estate.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression-Analysis/Linear-Regression-Real-Estate.ipynb --output Linear-Regression-Real-Estate.ipynb --output-dir 01-Regression-Analysis
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
