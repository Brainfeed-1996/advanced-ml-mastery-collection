# XGBoost - Customer Churn Prediction

**Folder:** `02-Classification`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Predict customer churn using gradient boosting (XGBoost)

## Key Features

- Customer behavior dataset analysis
- Feature engineering for churn prediction
- XGBoost model training and tuning
- Early stopping for prevention of overfitting
- Model interpretation with SHAP values

## How to Use

Run the notebook to train a churn prediction model. Use SHAP for model interpretation.

## Expected Outputs

Trained XGBoost model, feature importance, SHAP plots, churn predictions

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 02-Classification/XGBoost-Customer-Churn.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification/XGBoost-Customer-Churn.ipynb --output XGBoost-Customer-Churn.ipynb --output-dir 02-Classification
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
