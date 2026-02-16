# BERT - Sentiment Analysis

**Folder:** `03-Clustering-Techniques`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Sentiment classification using BERT transformer model

## Key Features

- Text preprocessing and tokenization
- BERT model fine-tuning
- Sentiment classification pipeline
- Model evaluation on test data
- Inference on new text

## How to Use

Run the notebook to train a sentiment classifier. Use the trained model for inference.

## Expected Outputs

Trained BERT model, accuracy metrics, sentiment predictions

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 03-Clustering-Techniques/BERT-Sentiment-Analysis.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering-Techniques/BERT-Sentiment-Analysis.ipynb --output BERT-Sentiment-Analysis.ipynb --output-dir 03-Clustering-Techniques
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
