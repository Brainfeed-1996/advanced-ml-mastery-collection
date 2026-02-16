# Prophet - Market Trends Forecasting

**Folder:** `06-Reinforcement-Learning`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Forecast market trends using Facebook Prophet

## Key Features

- Time series data preparation
- Prophet model configuration
- Trend and seasonality analysis
- Forecast generation
- Model evaluation

## How to Use

Run the notebook to generate forecasts. Adjust seasonality parameters.

## Expected Outputs

Trend forecasts, seasonality plots, prediction intervals

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 06-Reinforcement-Learning/Prophet-Market-Trends.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/Prophet-Market-Trends.ipynb --output Prophet-Market-Trends.ipynb --output-dir 06-Reinforcement-Learning
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
