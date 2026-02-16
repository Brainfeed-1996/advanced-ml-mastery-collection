# ResNet - Transfer Learning

**Folder:** `05-Computer-Vision`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Transfer learning with ResNet for image classification

## Key Features

- Pre-trained ResNet model loading
- Transfer learning implementation
- Fine-tuning strategies
- Feature extraction
- Model evaluation

## How to Use

Run the notebook to apply transfer learning. Use pre-trained weights for faster training.

## Expected Outputs

Fine-tuned ResNet model, accuracy metrics, prediction examples

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 05-Computer-Vision/ResNet-Transfer-Learning.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  05-Computer-Vision/ResNet-Transfer-Learning.ipynb --output ResNet-Transfer-Learning.ipynb --output-dir 05-Computer-Vision
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
