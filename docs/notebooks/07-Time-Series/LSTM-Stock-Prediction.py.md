# LSTM - Stock Price Prediction

**Folder:** `07-Time-Series`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Predict stock prices using LSTM neural networks

## Key Features

- Time series data preprocessing
- LSTM architecture design
- Sequence modeling for prediction
- Model training with callbacks
- Forecasting future prices

## How to Use

Run the notebook to train an LSTM model. Use it for stock price forecasting.

## Expected Outputs

Trained LSTM model, prediction plots, forecast metrics

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 07-Time-Series/LSTM-Stock-Prediction.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series/LSTM-Stock-Prediction.ipynb --output LSTM-Stock-Prediction.ipynb --output-dir 07-Time-Series
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
