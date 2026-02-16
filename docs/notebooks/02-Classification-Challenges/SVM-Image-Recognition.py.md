# SVM - Image Recognition

**Folder:** `02-Classification-Challenges`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Image classification using Support Vector Machines

## Key Features

- Image feature extraction
- SVM for image classification
- Data augmentation techniques
- Model evaluation on test set
- Visualization of predictions

## How to Use

Run the notebook to train an image classifier. Use data augmentation for better performance.

## Expected Outputs

Trained SVM model, accuracy metrics, prediction examples

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 02-Classification-Challenges/SVM-Image-Recognition.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification-Challenges/SVM-Image-Recognition.ipynb --output SVM-Image-Recognition.ipynb --output-dir 02-Classification-Challenges
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
