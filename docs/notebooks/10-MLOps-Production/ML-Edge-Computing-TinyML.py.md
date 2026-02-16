# ML Edge Computing - TinyML

**Folder:** `10-MLOps-Production`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Deploy ML models on edge devices using TinyML techniques

## Key Features

- Model compression techniques
- Edge device deployment
- TinyML optimization
- Performance evaluation
- Real-world applications

## How to Use

Run the notebook to prepare models for edge deployment. Use optimized models on devices.

## Expected Outputs

Optimized models, deployment scripts, performance metrics

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 10-MLOps-Production/ML-Edge-Computing-TinyML.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  10-MLOps-Production/ML-Edge-Computing-TinyML.ipynb --output ML-Edge-Computing-TinyML.ipynb --output-dir 10-MLOps-Production
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
