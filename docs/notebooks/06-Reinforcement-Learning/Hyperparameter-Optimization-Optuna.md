# Hyperparameter Optimization - Optuna

**Folder:** `06-Reinforcement-Learning`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Optimize hyperparameters using Optuna framework

## Key Features

- Optuna study setup
- Hyperparameter search space definition
- Pruning strategies
- Best parameters extraction
- Visualization of optimization

## How to Use

Run the notebook to optimize hyperparameters. Use results for model training.

## Expected Outputs

Best hyperparameters, optimization history, parameter importance

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 06-Reinforcement-Learning/Hyperparameter-Optimization-Optuna.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/Hyperparameter-Optimization-Optuna.ipynb --output Hyperparameter-Optimization-Optuna.ipynb --output-dir 06-Reinforcement-Learning
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
