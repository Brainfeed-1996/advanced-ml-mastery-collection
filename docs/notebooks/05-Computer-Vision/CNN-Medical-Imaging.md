# CNN - Medical Imaging Diagnosis

**Folder:** `05-Computer-Vision`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Medical image classification using Convolutional Neural Networks

## Key Features

- Medical image preprocessing
- CNN architecture for medical imaging
- Transfer learning with pre-trained models
- Model evaluation on medical data
- Interpretability techniques

## How to Use

Run the notebook to train a medical image classifier. Use transfer learning for better performance.

## Expected Outputs

Trained CNN model, accuracy metrics, prediction examples

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 05-Computer-Vision/CNN-Medical-Imaging.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  05-Computer-Vision/CNN-Medical-Imaging.ipynb --output CNN-Medical-Imaging.ipynb --output-dir 05-Computer-Vision
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
